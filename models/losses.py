import torch.nn as nn
import torch.nn.functional as F


def image_loss(host, watermarked):
    return F.mse_loss(watermarked, host)


bce_loss = nn.BCEWithLogitsLoss()

def watermark_loss(pred_wm, true_wm):
    return bce_loss(pred_wm, true_wm)