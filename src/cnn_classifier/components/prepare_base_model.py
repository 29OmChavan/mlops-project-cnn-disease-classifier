"""
Stage 02: Base Model & Transfer Learning Component
Implements:
1. FirstPrinciplesResidualBlock (demonstrating Conv2d + BatchNorm2d + ReLU + Residual Skip Connection H(x) = F(x) + x)
2. Pretrained ResNet-18 Transfer Learning model with configurable layer freezing (FREEZE_ALL / FREEZE_TILL)
   and a regularized custom classification head.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.entity.config_entity import PrepareBaseModelConfig


def conv2d_from_scratch(
    image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
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
            patch = image[i * stride:i * stride + k_h, j * stride:j * stride + k_w]
            out[i, j] = np.sum(patch * kernel)
    return out


class FirstPrinciplesResidualBlock(nn.Module):
    """
    First-Principles Residual Convolutional Block (Andrew Ng / He et al. ResNet formulation):
        y = ReLU(BatchNorm(W_2 * ReLU(BatchNorm(W_1 * x))) + x)
    Included for educational experiments and custom head feature refinement.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv_block(x)
        out = out + identity  # Skip connection prevents vanishing gradients
        return self.relu(out)


class DiseaseClassifierTransferNet(nn.Module):
    """
    Production Transfer Learning CNN built on ResNet-18 backbone with layer freezing
    and a custom regularized multi-class classification head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_all: bool = False,
        freeze_till: int = 4,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception as dl_err:
            logger.warning(
                f"Pretrained weights download skipped ({dl_err}); initializing ResNet-18 weights."
            )
            backbone = models.resnet18(weights=None)

        # Extract feature extractor children modules:
        # [0] conv1, [1] bn1, [2] relu, [3] maxpool, [4] layer1, [5] layer2, [6] layer3, [7] layer4, [8] avgpool
        children = list(backbone.children())[:-1]
        self.features = nn.Sequential(*children)

        # Apply Transfer Learning layer freezing policy
        if freeze_all:
            for param in self.features.parameters():
                param.requires_grad = False
        elif freeze_till > 0:
            for idx, child in enumerate(self.features.children()):
                if idx < freeze_till:
                    for param in child.parameters():
                        param.requires_grad = False

        in_features = backbone.fc.in_features  # 512 for ResNet-18
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class PrepareBaseModel:
    """Builds and saves the base and transfer-learning-updated PyTorch CNN architectures."""

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self) -> nn.Module:
        try:
            base_model = DiseaseClassifierTransferNet(
                num_classes=self.config.params_classes,
                pretrained=(self.config.params_weights == "DEFAULT"),
                freeze_all=True,
                freeze_till=0,
                dropout_rate=self.config.params_dropout_rate,
            )
            torch.save(base_model.state_dict(), self.config.base_model_path)
            logger.info(f"Saved frozen base model state_dict at: {self.config.base_model_path}")
            return base_model
        except Exception as e:
            raise CustomException(e, sys)

    def update_base_model(self) -> nn.Module:
        try:
            updated_model = DiseaseClassifierTransferNet(
                num_classes=self.config.params_classes,
                pretrained=(self.config.params_weights == "DEFAULT"),
                freeze_all=self.config.params_freeze_all,
                freeze_till=self.config.params_freeze_till,
                dropout_rate=self.config.params_dropout_rate,
            )

            total_params = sum(p.numel() for p in updated_model.parameters())
            trainable_params = sum(
                p.numel() for p in updated_model.parameters() if p.requires_grad
            )
            logger.info(
                f"Prepared Transfer Learning Model: Total params={total_params:,} | "
                f"Trainable params={trainable_params:,} ({100.0 * trainable_params / total_params:.1f}%)"
            )

            torch.save(updated_model.state_dict(), self.config.updated_base_model_path)
            logger.info(
                f"Saved updated transfer learning model at: {self.config.updated_base_model_path}"
            )
            return updated_model
        except Exception as e:
            raise CustomException(e, sys)
