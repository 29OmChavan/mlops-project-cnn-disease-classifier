"""
Stage 03: Model Training Component with Data Augmentation, Callbacks, and Dynamic Quantization.
Implements:
- Data Augmentation Pipeline (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter)
- Callbacks: ModelCheckpoint (saves best val_acc weights), EarlyStopping, CosineAnnealingLR
- Post-Training Dynamic Quantization (FP32 -> INT8 linear layers) for low-latency CPU inference
"""
import sys
import time
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.utils.common import save_json
from src.cnn_classifier.entity.config_entity import TrainingConfig
from src.cnn_classifier.components.prepare_base_model import DiseaseClassifierTransferNet


class ModelTraining:
    """Executes the augmented CNN training loop with checkpointing, early stopping, and INT8 quantization."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataloaders(self):
        img_h, img_w = self.config.params_image_size[0], self.config.params_image_size[1]
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        if self.config.params_is_augmentation:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((img_h, img_w)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((img_h, img_w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=str(self.config.training_data / "train"), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=str(self.config.training_data / "val"), transform=val_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader, train_dataset.classes

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting CNN Training on device: {self.device}")
        try:
            torch.manual_seed(42)
            train_loader, val_loader, class_names = self.get_dataloaders()

            model = DiseaseClassifierTransferNet(
                num_classes=self.config.params_classes,
                pretrained=False,
                freeze_all=False,
                freeze_till=4,
            )
            if self.config.updated_base_model_path.exists():
                state_dict = torch.load(
                    self.config.updated_base_model_path, map_location=self.device, weights_only=True
                )
                model.load_state_dict(state_dict)
                logger.info(
                    f"Loaded updated base model weights from {self.config.updated_base_model_path}"
                )

            model = model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=self.config.params_learning_rate, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.params_epochs
            )

            best_val_acc = -1.0
            best_val_loss = float("inf")
            patience_counter = 0
            history: Dict[str, List[float]] = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "lr": [],
            }

            start_time = time.time()
            for epoch in range(1, self.config.params_epochs + 1):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_train += int((preds == labels).sum().item())
                    total_train += inputs.size(0)

                epoch_train_loss = running_loss / max(total_train, 1)
                epoch_train_acc = correct_train / max(total_train, 1)

                # Validation phase
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        correct_val += int((preds == labels).sum().item())
                        total_val += inputs.size(0)

                epoch_val_loss = val_loss / max(total_val, 1)
                epoch_val_acc = correct_val / max(total_val, 1)
                current_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step()

                history["train_loss"].append(round(epoch_train_loss, 4))
                history["train_acc"].append(round(epoch_train_acc, 4))
                history["val_loss"].append(round(epoch_val_loss, 4))
                history["val_acc"].append(round(epoch_val_acc, 4))
                history["lr"].append(round(current_lr, 6))

                logger.info(
                    f"Epoch [{epoch:02d}/{self.config.params_epochs:02d}] "
                    f"Train Loss={epoch_train_loss:.4f} Acc={epoch_train_acc:.4f} | "
                    f"Val Loss={epoch_val_loss:.4f} Acc={epoch_val_acc:.4f}"
                )

                # ModelCheckpoint callback logic
                if (epoch_val_acc > best_val_acc) or (
                    epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss
                ):
                    best_val_acc = epoch_val_acc
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "classes": class_names,
                            "image_size": self.config.params_image_size,
                            "best_val_acc": round(best_val_acc, 4),
                            "best_val_loss": round(best_val_loss, 4),
                        },
                        self.config.trained_model_path,
                    )
                    logger.info(
                        f"ModelCheckpoint: Saved new best model at {self.config.trained_model_path} "
                        f"(val_acc={best_val_acc:.4f})"
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.params_early_stopping_patience:
                        logger.info(
                            f"EarlyStopping triggered at epoch {epoch} (patience={self.config.params_early_stopping_patience})"
                        )
                        break

            elapsed_sec = round(time.time() - start_time, 2)

            # Novel Extension: Export Dynamically Quantized INT8 model for edge/CPU inference
            checkpoint = torch.load(
                self.config.trained_model_path, map_location="cpu", weights_only=False
            )
            best_model_cpu = DiseaseClassifierTransferNet(
                num_classes=self.config.params_classes, pretrained=False
            )
            best_model_cpu.load_state_dict(checkpoint["model_state_dict"])
            best_model_cpu.eval()

            quantized_model = torch.ao.quantization.quantize_dynamic(
                best_model_cpu, {nn.Linear}, dtype=torch.qint8
            )
            torch.save(
                {
                    "quantized_state_dict": quantized_model.state_dict(),
                    "classes": class_names,
                    "image_size": self.config.params_image_size,
                },
                self.config.quantized_model_path,
            )
            logger.info(
                f"Saved dynamically quantized INT8 model at: {self.config.quantized_model_path}"
            )

            training_summary = {
                "best_val_acc": round(best_val_acc, 4),
                "best_val_loss": round(best_val_loss, 4),
                "epochs_completed": len(history["train_loss"]),
                "training_duration_sec": elapsed_sec,
                "classes": class_names,
                "history": history,
            }
            save_json(self.config.training_history_path, training_summary)
            return training_summary

        except Exception as e:
            raise CustomException(e, sys)
