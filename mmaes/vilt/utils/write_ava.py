import pandas as pd
import pyarrow as pa
import numpy as np
import os


def make_arrow(root, dataset_root):
    splits = ["train", "val", "test"]
    image_root = "/root/autodl-tmp/project/datasets/AVA/images"
    text_root = "/root/autodl-tmp/project/datasets/AVA/texts"

    for split in splits:
        df = pd.read_csv(f"{root}/{split}.csv")
        data_list = []
        missing_files = []
        for row in df.itertuples():
            label = 0
            image_id = getattr(row, "image_id")
            image_name = str(image_id) + ".jpg"
            image_path = os.path.join(image_root, image_name)
            try:
                with open(image_path, "rb") as fp:
                    binary = fp.read()
            except FileNotFoundError:
                missing_files.append(image_name)
                continue

            score_lst = []
            for i in range(10):
                score_idx = i+1
                score = getattr(row, "score"+str(score_idx))
                score_lst.append(score)
            score_lst = np.array(score_lst)
            label = score_lst / score_lst.sum()

            text_name = str(image_id) + ".txt"
            text_path = os.path.join(text_root, text_name)
            try:
                with open(text_path, "r") as fp:
                    text_ = fp.read()
                    text = [text_]
            except UnicodeError:
                missing_files.append(text_name)

            data = (binary, text, label, image_id, split)
            data_list.append(data)

        print(missing_files)
        f = open("skipping_file.txt", "w")
        for line in missing_files:
            f.write(line + '\n')
        f.close()

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/ava_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

