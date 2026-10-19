"""
Real-Time Computer Vision Inference Pipeline.
Loads trained ResNet-18 disease classifier weights, transforms uploaded/base64 images,
and returns predicted disease class, confidence probability, per-class probabilities, and clinical guidance.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import torch
from torchvision import transforms

from src.cnn_classifier.logger import logger
from src.cnn_classifier.exception import CustomException
from src.cnn_classifier.components.prepare_base_model import DiseaseClassifierTransferNet

CLINICAL_GUIDANCE = {
    "Coccidiosis": "Parasitic protozoan infection (Eimeria). Isolate affected flock, administer anticoccidial treatment (amprolium/toltrazuril), and sanitize litter moisture.",
    "Healthy": "No pathological biomarkers detected. Maintain standard biosecurity, vaccination schedule, and balanced feed hydration.",
    "Newcastle_Disease": "Highly contagious avian paramyxovirus (NDV). Immediate quarantine required; notify veterinary authority and enforce strict biosecurity.",
    "Salmonella": "Bacterial enteric infection (Salmonella enterica). Conduct bacteriological culture confirmation, sanitize water lines, and apply targeted antimicrobial protocol.",
}


class PredictionPipeline:
    """Runs single-image inference on microscopy/pathology images."""

    def __init__(
        self,
        filename: Path,
        model_path: Path = Path("artifacts/training/model.pt"),
    ):
        self.filename = Path(filename)
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self) -> Dict[str, Any]:
        try:
            t0 = time.perf_counter()
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )
            classes = checkpoint.get(
                "classes", ["Coccidiosis", "Healthy", "Newcastle_Disease", "Salmonella"]
            )
            img_size = checkpoint.get("image_size", [64, 64, 3])

            model = DiseaseClassifierTransferNet(
                num_classes=len(classes), pretrained=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            preprocess = transforms.Compose(
                [
                    transforms.Resize((img_size[0], img_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            img = Image.open(self.filename).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx = int(probs.argmax())
            pred_label = classes[pred_idx]
            confidence = round(float(probs[pred_idx]) * 100.0, 2)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            prob_dict = {
                cls_name: round(float(probs[i]) * 100.0, 2)
                for i, cls_name in enumerate(classes)
            }

            logger.info(
                f"Predicted '{pred_label}' ({confidence}%) for {self.filename} in {latency_ms}ms"
            )

            return {
                "image": str(self.filename.name),
                "predicted_class": pred_label,
                "confidence_percent": confidence,
                "class_probabilities_percent": prob_dict,
                "clinical_guidance": CLINICAL_GUIDANCE.get(pred_label, ""),
                "inference_latency_ms": latency_ms,
            }
        except Exception as e:
            raise CustomException(e, sys)
