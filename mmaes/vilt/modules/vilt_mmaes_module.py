import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import vilt.modules.vision_transformer_mmaes as vit
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr, pearsonr
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from vilt.gadgets.my_metrics import get_score


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        # self.mma = MissModalityAlign(config["hidden_size"], config["hidden_size"])
        self.vis = config["visual"]
        self.vis_embeddings = torch.zeros((0, 768), dtype=torch.float32)
        self.vis_predict = []

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"]
                and not self.hparams.config["finetune_first"]
        ):
            #
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,
                                                                                                                     -1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1, 1, 40, 768),
                                                          size=(config["max_text_len"], 768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["ava"] > 0:
            cls_num = self.hparams.config["ava_class_num"]
            self.ava_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
                nn.Softmax(dim=1)
            )
            self.ava_classifier.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("use pre-finetune model")

        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_pandd = self.hparams.config["multi_layer_pandd"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_pandd else 1

        digest_length = self.hparams.config["digest_length"]
        self.digest_length = digest_length
        self.digest_layers = self.hparams.config["digest_layers"]

        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:, 0:1, :].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:, 2:3, :].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:, 1:2, :].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        if not self.learnt_p:
            self.complete_prompt.requires_grad = False
            self.missing_text_prompt.requires_grad = False
            self.missing_img_prompt.requires_grad = False

        # print(self.complete_prompt)
        # print(self.missing_img_prompt)
        # print(self.missing_text_prompt)

        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", nn.Conv1d(768, 768, 3))
        to_global_feature.add_module("pool", nn.AdaptiveAvgPool1d(digest_length))
        self.to_global_feature = to_global_feature

        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.text_embeddings.parameters():
            param.requires_grad = False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad = False

        vilt_utils.set_metrics(self)
        self.pred_score = []
        self.true_score = []
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
            is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )


        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        # build missing-modal prompts
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt
            elif batch["missing_type"][idx] == 1:
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_img_prompt

            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)

            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)

        # build missing-modal digests
        digests = None
        for i in self.digest_layers:
            d_prompts = None
            for idx in range(len(img)):
                d_txt = text_embeds[idx]
                d_img = image_embeds[idx]
                if prompts is None:
                    d_pmt = torch.empty(0, d_txt.shape[1], dtype=d_txt.dtype, device=d_txt.device)
                else:
                    d_pmt = prompts[idx, self.digest_layers.index(i)]
                if batch["missing_type"][idx] == 0:
                    dgt = torch.cat([d_pmt, d_txt, d_img], dim=0).unsqueeze(0).transpose(1, 2)
                    d_prompt = self.to_global_feature(dgt).transpose(1, 2).unsqueeze(0)
                elif batch["missing_type"][idx] == 1:
                    dgt = torch.cat([d_pmt, d_txt], dim=0).unsqueeze(0).transpose(1, 2)
                    d_prompt = self.to_global_feature(dgt).transpose(1, 2).unsqueeze(0)
                elif batch["missing_type"][idx] == 2:
                    dgt = torch.cat([d_pmt, d_img], dim=0).unsqueeze(0).transpose(1, 2)
                    d_prompt = self.to_global_feature(dgt).transpose(1, 2).unsqueeze(0)
                if d_prompts is None:
                    d_prompts = d_prompt
                else:
                    d_prompts = torch.cat([d_prompts, d_prompt], dim=1)
            if digests is None:
                digests = d_prompts
            else:
                digests = torch.cat([digests, d_prompts], dim=0)
        digests = digests.transpose(0, 1)

        if prompts is not None:
            if self.learnt_p:
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length * len(self.prompt_layers),
                                          dtype=prompts.dtype, device=prompts.device).long()
            else:
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype,
                                          device=prompts.device).long()
        else:
            prompt_masks = torch.empty(text_masks.shape[0], 0, dtype=text_masks.dtype, device=text_masks.device).long()

        if digests is None:
            d_masks = torch.empty(prompts.shape[0], 0, dtype=prompts.dtype,
                                  device=prompts.device).long()
        else:
            d_masks = torch.ones(text_masks.shape[0], self.digest_length, dtype=text_masks.dtype,
                                 device=text_masks.device).long()
        co_masks = torch.cat([prompt_masks, text_masks, image_masks, d_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)

        # common space projection
        # co_embeds_pro = self.mma(co_embeds)
        # x = co_embeds_pro.detach()
        # co_embeds = self.mma(text_embeds, image_embeds)
        x = co_embeds.detach()

        input_data = x

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_pandd:
                    if digests is not None and prompts is not None:
                        x, _attn = blk(x, mask=co_masks,
                                       prompts=prompts[:, self.prompt_layers.index(i)],
                                       digests=digests[:, self.digest_layers.index(i)],
                                       learnt_p=self.learnt_p)
                    elif digests is None:
                        x, _attn = blk(x, mask=co_masks,
                                       prompts=prompts[:, self.prompt_layers.index(i)],
                                       learnt_p=self.learnt_p)
                    elif prompts is None:
                        x, _attn = blk(x, mask=co_masks,
                                       digests=digests[:, self.digest_layers.index(i)],
                                       learnt_p=self.learnt_p)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, digests=digests, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)

        if prompts is not None:
            total_prompt_len = len(self.prompt_layers) * prompts.shape[-2]
        else:
            total_prompt_len = 0

        text_feats, image_feats = (
            x[:, total_prompt_len: total_prompt_len + text_embeds.shape[1]],
            x[:, total_prompt_len + text_embeds.shape[1]:],
        )
        cls_feats = self.pooler(x[:, total_prompt_len:total_prompt_len + 1])
        # cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret, input_data

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            model_ret, _ = self.infer(batch)
            ret.update(model_ret)
            return ret, _

        # Classification for AVA
        if "ava" in self.current_tasks:
            ava_ret, _ = objectives.compute_ava(self, batch)
            ret.update(ava_ret)
        return ret, _

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output, data = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output, data = self(batch)

        labels = output['ava_labels']
        logits = output['ava_logits']
        pscore, pscore_np = get_score(logits)
        tscore, tscore_np = get_score(labels)
        self.pred_score = pscore_np.tolist() + self.pred_score
        self.true_score = tscore_np.tolist() + self.true_score

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        acc = accuracy_score((np.where(np.array(self.pred_score) <= 5.00, 0, 1)),
                             np.where(np.array(self.true_score) <= 5.00, 0, 1))
        srcc_mean = spearmanr(self.pred_score, self.true_score)
        lcc_mean = pearsonr(self.pred_score, self.true_score)
        self.pred_score = []
        self.true_score = []

        print(acc)  # 自己打印
        print(srcc_mean)
        print(lcc_mean)
        f = open('/root/autodl-tmp/project/val_log.txt', 'a')
        f.write('acc:%.4f,srcc:%.4f,lcc:%.4f\r\n'
                % (acc, srcc_mean[0], lcc_mean[0]))
        f.flush()

    #         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
    #         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
    #         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output, data = self(batch)

        if self.vis:
            for embed in output["embeddings"]:
                embed = embed.unsqueeze(0)
                self.vis_embeddings = torch.cat((self.vis_embeddings, embed.detach().cpu()), 0)
            self.vis_predict.extend(output["pred"])
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        if self.vis:
            self.vis_embeddings = np.array(self.vis_embeddings)
            self.vis_predict = np.array(self.vis_predict)

            tsne = TSNE(2, verbose=1)
            tsne_proj = tsne.fit_transform(self.vis_embeddings)

            # pca = PCA(n_components=2)
            # pca.fit(self.vis_embeddings)
            # pca_proj = pca.transform(self.vis_embeddings)
            # pca.explained_variance_ratio_


            cmap = cm.get_cmap('tab20')
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(2):
                indices = self.vis_predict == i
                ax.scatter(tsne_proj[indices, 0],
                           tsne_proj[indices, 1],
                           c=np.array(cmap(i)).reshape(1, 4),
                           label=i,
                           alpha=0.5)

                # ax.scatter(pca_proj[indices, 0],
                #            pca_proj[indices, 1],
                #            c=np.array(cmap(i)).reshape(1, 4),
                #            label=i,
                #            alpha=0.5)

            ax.legend(fontsize='large', markerscale=2)
            plt.show()
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

