import torch


def get_criterion(pred: torch.Tensor, target: torch.Tensor):
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    # reduction="none": 각 data point에 대한 손실 개별적으로 계산 후 tensor 반환
    return loss(pred, target)
