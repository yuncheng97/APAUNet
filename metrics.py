import numpy as np
import torch
from medpy import metric
import torch.nn.functional as F

def dice_score(output, target):
    """
    Calculate the Dice score for the given output and target.

    Args:
        output (torch.Tensor or np.ndarray): The predicted segmentation output.
        target (torch.Tensor or np.ndarray): The ground truth segmentation target.

    Returns:
        np.ndarray: A 2D array containing the Dice scores for each class and sample.
    """
    smooth = 1e-4
    if torch.is_tensor(output):
        output = torch.nn.Softmax(dim=1)(output)
        output = torch.argmax(output, 1)
        output = output.data.cpu().numpy()

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dice = [[], [], []]
    for idx in range(output.shape[0]):
        for i in range(3):
            intersection = ((output[idx] == i) & (target[idx] == i)).sum()
            union = ((output[idx] == i) | (target[idx] == i)).sum()
            cof = (intersection + smooth) / (union + smooth)
            dice[i].append(2 * cof / (1 + cof))
    return np.asarray(dice)

def load_hdfunc(outputs, targets):
    """
    Combine the outputs using weights and add a new dimension.

    Args:
        outputs (torch.Tensor): The predicted segmentation outputs with shape (batch_size, num_classes, height, width).
        targets (torch.Tensor): The ground truth segmentation targets with shape (batch_size, 1, height, width).

    Returns:
        torch.Tensor: The combined outputs with shape (batch_size, 1, height, width).
    """
    output = (outputs[:, 0] * 0) + (outputs[:, 1] * 1) + (outputs[:, 2] * 2)
    output = output.unsqueeze(1)

    # Calculate the Hausdorff distance (hd95) between output and targets
    hd95 = metric.binary.hd95(output.detach().cpu().numpy(), targets.detach().cpu().numpy())

    return hd95