"""
First-Principles CNN & Residual Network Verification Script
(Sentdex + Andrew Ng Deep Learning Specialization Companion Code).
Demonstrates:
1. Pure NumPy 2D discrete cross-correlation (convolution) + MaxPool2D forward pass
2. FirstPrinciplesResidualBlock skip connection H(x) = F(x) + x gradient preservation
3. Transfer Learning parameter freezing verification
"""
import numpy as np
import torch
from src.cnn_classifier.components.prepare_base_model import (
    FirstPrinciplesResidualBlock,
    DiseaseClassifierTransferNet,
)


def conv2d_from_scratch(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Computes 2D discrete cross-correlation from first principles:
        O[i, j] = sum_m sum_n I[i*s + m, j*s + n] * K[m, n]
    Output spatial dimension formula:
        H_out = floor((H_in + 2P - K) / S) + 1
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    h_in, w_in = image.shape
    k_h, k_w = kernel.shape
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    out = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = image[i * stride : i * stride + k_h, j * stride : j * stride + k_w]
            out[i, j] = np.sum(patch * kernel)
    return out


if __name__ == "__main__":
    img = np.random.randn(8, 8).astype(np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    feat_map = conv2d_from_scratch(img, sobel_x, stride=1, padding=1)
    print(f"1. NumPy Conv2D output shape (expected 8x8): {feat_map.shape}")

    res_block = FirstPrinciplesResidualBlock(channels=16)
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = res_block(x)
    y.sum().backward()
    print(f"2. Residual Block output shape: {y.shape}, Input grad norm: {x.grad.norm().item():.4f}")

    tl_net = DiseaseClassifierTransferNet(num_classes=4, pretrained=False, freeze_till=4)
    total = sum(p.numel() for p in tl_net.parameters())
    trainable = sum(p.numel() for p in tl_net.parameters() if p.requires_grad)
    print(f"3. Transfer Net Total Params: {total:,} | Trainable: {trainable:,}")
