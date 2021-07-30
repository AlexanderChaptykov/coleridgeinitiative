import copy
from typing import Set, Dict, Union, List

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import

from pytorch_lightning.metrics.utils import class_reduce
from torch.nn import Module
from torch.nn import functional as F
from pytorch_lightning.loggers import CometLogger


def weighted_cross(preds, targets):
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 20]))
    return criterion(preds.type(torch.float32), targets[:, 1].type(torch.long))


class F1Loss(Module):
    """
    Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)

        # we already have log_softmax values, probs = exp(log_softmax(logits)) = softmax(logits)
        # log softmax is more stable than softmax
        y_pred = torch.exp(y_pred)
        # y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, target):
        ce_loss = F.nll_loss(preds, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class PrecisionRecallF1(Metric):
    """
    Calculates precision, recall and F1 score for each class independently.
    """

    def __init__(self, num_classes=1, compute_on_step=False, dist_sync_on_step=True, padding_idx=None,
                 reset_with_compute=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.padding_idx = padding_idx
        self.reset_with_compute = reset_with_compute
        self.num_classes = num_classes
        self.add_state("tps", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("fps", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("tns", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("fns", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("sups", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.padding_idx is not None:
            mask = target != self.padding_idx
            preds = preds.masked_select(mask)
            target = target.masked_select(mask)

        tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred=preds, target=target,
                                                                num_classes=self.num_classes)
        self.tps += tps
        self.fps += fps
        self.tns += tns
        self.fns += fns
        self.sups += sups

    def compute(self):
        precision = class_reduce(self.tps, self.tps + self.fps, self.sups, class_reduction='none')
        recall = class_reduce(self.tps, self.tps + self.fns, self.sups, class_reduction='none')
        f1_score = (2 * precision * recall) / (precision + recall)
        if self.reset_with_compute:
            self.reset()
        return precision, recall, f1_score


def ce_loss(pred, truth, smoothing=False, trg_pad_idx=-1, eps=0.1
            ):
    """
    Computes the cross entropy loss with label smoothing

    Args:
        pred (torch tensor): Prediction
        truth (torch tensor): Target
        smoothing (bool, optional): Whether to use smoothing. Defaults to False.
        trg_pad_idx (int, optional): Indices to ignore in the loss. Defaults to -1.
        eps (float, optional): Smoothing coefficient. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    truth = truth.contiguous().view(-1)  # nothing changed
    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)

    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    loss = -one_hot * F.log_softmax(pred, dim=1)

    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)

    return loss.sum(axis=1)


def qa_loss_fn(
        start_logits,
        end_logits,
        start_positions,
        end_positions,
        labels: List[str],
        label_w=1,
        non_label_w=.01,
        config={"smoothing": False, "eps": 0.1}
):
    """
    Loss function for the question answering task.
    It is the sum of the cross entropy for the start and end logits

    Args:
        start_logits (torch tensor): Start logits
        end_logits (torch tensor): End logits
        start_positions (torch tensor): Start ground truth
        end_positions (torch tensor): End ground truth
        config (dict): Dictionary of parameters for the CE Loss.

    Returns:
        torch tensor: Loss value

        qa_loss_fn(out[0], out[1], torch.tensor([3,3,3], dtype=torch.int64), torch.tensor([4,4,4], dtype=torch.int64), config={
                "smoothing": False,
                "eps": 0.1,
            })
    """
    bs = start_logits.size(0)
    mask = torch.tensor([label_w if x else non_label_w for x in labels], dtype=start_logits.dtype)
    mask = mask.to(start_logits.device)
    # todo заменить на конфиги проекта
    start_loss = ce_loss(
        start_logits,
        start_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
    )

    end_loss = ce_loss(
        end_logits,
        end_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
    )

    total_loss = start_loss + end_loss
    total_loss = torch.matmul(total_loss, mask)

    return total_loss / bs


from typing import List
import numpy as np


# noinspection PyTypeChecker
def compute_fbeta(y_true: List[Set[str]],
                  y_pred: List[Set[str]],
                  beta: float = 0.5) -> Dict[str, Union[float, int]]:
    """Compute the Jaccard-based micro FBeta score.

    References
    ----------
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    """
    assert len(y_true) == len(y_pred)
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)

    def _jaccard_similarity(str1: str, str2: str) -> float:
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        if (len(a) + len(b) - len(c)) != 0:
            return float(len(c)) / (len(a) + len(b) - len(c))
        return 0

    tp = 0  # найденные таргеты
    fp = 0  # ошибочные предсказания
    fn = 0  # не найденные таргеты
    for ground_truth_list, predicted_string_list in zip(y_true, y_pred):
        for ground_truth in ground_truth_list:
            if len(predicted_string_list) == 0:
                fn += 1
            else:
                similarity_scores = [
                    _jaccard_similarity(ground_truth, predicted_string)
                    for predicted_string in predicted_string_list
                ]
                matched_idx = np.argmax(similarity_scores)
                if similarity_scores[matched_idx] >= 0.5:
                    predicted_string_list.pop()
                    tp += 1
                else:
                    fn += 1
        fp += len(predicted_string_list)

    tpr = tp / (tp + fn)

    tp_beta = tp * (1 + beta ** 2)
    fn_beta = fn * (beta ** 2)
    fbeta_score = tp_beta / (tp_beta + fp + fn_beta)
    return {'f1beta': fbeta_score, 'tpr': tpr, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / (tp + fp)}


def compute_fbeta_my(y_true: List[List[str]],
                     y_pred: List[List[str]],
                     beta: float = 0.5) -> float:
    """Compute the Jaccard-based micro FBeta score.

    References
    ----------
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    """


y = [{'asd', 'asd2', 'asd3'}, {'asd3', 'asd1', 'asd0'}]
pred = [{'asd', 'asd2', 'asd3'}, {'asd3', 'asd1', 'asd0'}]
compute_fbeta(y, pred)
