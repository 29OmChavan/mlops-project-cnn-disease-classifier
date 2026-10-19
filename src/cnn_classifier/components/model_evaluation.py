"""
Stage 04: Model Evaluation & MLflow Tracking Component
Computes multi-class metrics (Accuracy, Macro F1, Weighted F1, per-class Precision/Recall/F1,
ROC-AUC, Confusion Matrix, FP32 vs INT8 latency benchmark), writes scores.json for DVC,
and logs runs to DagsHub MLflow.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import mlflow
import mlflow.pytorch

from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.utils.common import save_json
from src.cnn_classifier.entity.config_entity import EvaluationConfig
from src.cnn_classifier.components.prepare_base_model import DiseaseClassifierTransferNet


class Evaluation:
    """Evaluates the trained CNN classifier, writes DVC scores.json, and logs to MLflow."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluation(self) -> Dict[str, Any]:
        logger.info("Entered Evaluation.evaluation()")
        try:
            img_h, img_w = self.config.params_image_size[0], self.config.params_image_size[1]
            val_transform = transforms.Compose(
                [
                    transforms.Resize((img_h, img_w)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            val_dataset = datasets.ImageFolder(
                root=str(self.config.training_data), transform=val_transform
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.params_batch_size,
                shuffle=False,
                num_workers=0,
            )

            checkpoint = torch.load(
                self.config.path_of_model, map_location=self.device, weights_only=False
            )
            num_classes = len(val_dataset.classes)
            model = DiseaseClassifierTransferNet(
                num_classes=num_classes, pretrained=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()

            criterion = nn.CrossEntropyLoss()
            total_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []

            t0 = time.perf_counter()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    total_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_probs.extend(probs.cpu().numpy().tolist())
            fp32_latency_ms = round(
                (time.perf_counter() - t0) * 1000.0 / max(len(val_dataset), 1), 3
            )

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)

            val_loss = float(total_loss / max(len(val_dataset), 1))
            acc = float(accuracy_score(y_true, y_pred))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
            weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
            roc_auc_ovr = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            cm = confusion_matrix(y_true, y_pred).tolist()

            prec, rec, f1_per_cls, _ = precision_recall_fscore_support(
                y_true, y_pred, zero_division=0
            )
            per_class_metrics = {}
            for idx, cls_name in enumerate(val_dataset.classes):
                per_class_metrics[cls_name] = {
                    "precision": round(float(prec[idx]), 4),
                    "recall": round(float(rec[idx]), 4),
                    "f1_score": round(float(f1_per_cls[idx]), 4),
                }

            # Benchmark INT8 dynamically quantized model latency
            quant_model = torch.ao.quantization.quantize_dynamic(
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            quant_model.eval()
            dummy_input = torch.randn(1, 3, img_h, img_w)
            t_q0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(20):
                    _ = quant_model(dummy_input)
            int8_latency_ms = round((time.perf_counter() - t_q0) * 1000.0 / 20.0, 3)

            fp32_size_mb = round(os.path.getsize(self.config.path_of_model) / (1024 * 1024), 2)
            int8_size_mb = (
                round(os.path.getsize(self.config.quantized_model_path) / (1024 * 1024), 2)
                if self.config.quantized_model_path.exists()
                else fp32_size_mb
            )

            self.score = {
                "loss": round(val_loss, 4),
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "roc_auc_ovr": round(roc_auc_ovr, 4),
                "fp32_inference_ms_per_image": fp32_latency_ms,
                "int8_quantized_inference_ms": int8_latency_ms,
                "fp32_model_size_mb": fp32_size_mb,
                "int8_model_size_mb": int8_size_mb,
                "per_class_metrics": per_class_metrics,
                "confusion_matrix": cm,
                "classes": val_dataset.classes,
            }

            save_json(path=self.config.score_file, data=self.score)
            save_json(path=Path("artifacts/evaluation/scores.json"), data=self.score)
            logger.info(
                f"Evaluation Complete: Accuracy={acc:.4f} | Macro F1={macro_f1:.4f} | ROC-AUC={roc_auc_ovr:.4f}"
            )
            return self.score

        except Exception as e:
            raise CustomException(e, sys)

    def log_into_mlflow(self) -> None:
        try:
            use_remote = bool(
                os.environ.get("MLFLOW_TRACKING_USERNAME")
                and os.environ.get("MLFLOW_TRACKING_PASSWORD")
                and os.environ.get("DISABLE_REMOTE_MLFLOW", "0") != "1"
            )
            active_uri = self.config.mlflow_uri if use_remote else "file:./mlruns"
            mlflow.set_tracking_uri(active_uri)
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name="resnet18_transfer_learning_eval"):
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(
                    {
                        "loss": self.score["loss"],
                        "accuracy": self.score["accuracy"],
                        "macro_f1": self.score["macro_f1"],
                        "weighted_f1": self.score["weighted_f1"],
                        "roc_auc_ovr": self.score["roc_auc_ovr"],
                        "fp32_inference_ms": self.score["fp32_inference_ms_per_image"],
                        "int8_inference_ms": self.score["int8_quantized_inference_ms"],
                    }
                )
                mlflow.log_artifact(str(self.config.score_file))
                logger.info(f"Logged evaluation metrics & artifacts to MLflow ({active_uri})")
        except Exception as mlf_err:
            logger.warning(f"MLflow logging warning: {mlf_err}")
