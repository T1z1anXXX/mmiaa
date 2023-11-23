import torch
import torch.nn as nn
from vilt.modules.vilt_utils import EMDLoss


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_ava(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer, input_data = pl_module.infer(batch, mask_text=False, mask_image=False)
    else:
        infer, input_data = pl_module.infer(batch, mask_text=False, mask_image=False)

    imgcls_logits = pl_module.ava_classifier(infer["cls_feats"])

    imgcls_labels = torch.tensor(batch["label"]).to(pl_module.device)
    # imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()


    criterion = EMDLoss()
    imgcls_loss = criterion(p_target=imgcls_labels, p_estimate=imgcls_logits)


    ret = {
        "ava_loss": imgcls_loss,
        "ava_logits": imgcls_logits,
        "ava_labels": imgcls_labels,
        "embeddings": infer["cls_feats"],
    }

    loss = getattr(pl_module, f"{phase}_ava_loss")(ret["ava_loss"])
    acc = getattr(pl_module, f"{phase}_ava_accuracy")(
        ret["ava_logits"], ret["ava_labels"]
    )
    ground, pred = getattr(pl_module, f"{phase}_ava_accuracy").get_label()
    srcc = getattr(pl_module, f"{phase}_ava_srcc")(
        ret["ava_logits"], ret["ava_labels"]
    )
    lcc = getattr(pl_module, f"{phase}_ava_lcc")(
        ret["ava_logits"], ret["ava_labels"]
    )
    pl_module.log(f"ava/{phase}/loss", loss)
    ret.update({"pred": pred})

    return ret, input_data
