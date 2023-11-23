import torch
from torchmetrics import Metric
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def get_score(y_pred):
    s = y_pred.size(1)
    w = torch.from_numpy(np.linspace(1, s, s))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class AVA_Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_score", default=[])
        self.add_state("pred_score", default=[])
        self.tsl = np.empty(0)
        self.psl = np.empty(0)

    def get_label(self):
        return self.tsl, self.psl

    def update(self, logits, labels):

        pscore, pscore_np = get_score(logits)
        tscore, tscore_np = get_score(labels)
        for p in pscore_np:
            self.pred_score.append(p)
        for t in tscore_np:
            self.true_score.append(t)

    def compute(self):
        true_score = np.array(self.true_score)

        true_score_lable = np.where(true_score <= 5.00, 0, 1)
        pred_score = np.array(self.pred_score)
        pred_score_lable = np.where(pred_score <= 5.00, 0, 1)
        self.tsl = true_score_lable
        self.psl = pred_score_lable
        return accuracy_score(true_score_lable, pred_score_lable)


class LCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_score", default=[])
        self.add_state("pred_score", default=[])
        # self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, labels):
        pscore, pscore_np = get_score(logits)
        tscore, tscore_np = get_score(labels)
        for p in pscore_np:
            self.pred_score.append(p)
        for t in tscore_np:
            self.true_score.append(t)

    def compute(self):
        PEARSONR = pearsonr(self.pred_score, self.true_score)
        lcc = torch.tensor(PEARSONR[0])
        return lcc


class SRCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_score", default=[])
        self.add_state("pred_score", default=[])
        # self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, labels):
        pscore, pscore_np = get_score(logits)
        tscore, tscore_np = get_score(labels)
        for p in pscore_np:
            self.pred_score.append(p)
        for t in tscore_np:
            self.true_score.append(t)

    def compute(self):
        SPEARMANR = spearmanr(self.pred_score, self.true_score)
        srcc = torch.tensor(SPEARMANR[0])
        return srcc

