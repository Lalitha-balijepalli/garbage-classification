"""
predict.py
==========
End-to-end single-image inference: load the best trained model once,
then classify any image and attach a recycling recommendation.

This is the module the Streamlit app imports — the app itself must not
contain any model or classification logic (keeps the UI layer thin and
the core system testable independently of Streamlit).
"""

import time

import torch
from PIL import Image, UnidentifiedImageError

from src import config
from src.models import build_model
from src.preprocessing import get_eval_transform
from src.recycling import get_recommendation
from src.utils import load_checkpoint


class GarbagePredictor:
    """Wraps a loaded model + its metadata for repeated inference calls."""

    def __init__(self, checkpoint_path: str = config.BEST_MODEL_PATH, device=config.DEVICE):
        self.device = device
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)

        self.class_names = checkpoint["class_names"]
        self.model_name = checkpoint["model_name"]
        self.image_size = checkpoint["image_size"]

        self.model = build_model(self.model_name, num_classes=len(self.class_names), freeze_backbone=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.transform = get_eval_transform(self.image_size)

    def predict_image(self, image_path: str, top_k: int = config.TOP_K,
                       confidence_threshold: float = config.CONFIDENCE_THRESHOLD) -> dict:
        """
        Classify a single image on disk and return a structured result
        (Section 27). Handles invalid/corrupted images gracefully instead
        of raising an unhandled exception (Section 29).
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            return {"error": f"Could not read image '{image_path}': {e}"}

        return self.predict_pil_image(image, top_k=top_k, confidence_threshold=confidence_threshold)

    def predict_pil_image(self, image: Image.Image, top_k: int = config.TOP_K,
                           confidence_threshold: float = config.CONFIDENCE_THRESHOLD) -> dict:
        start = time.time()

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        inference_time_ms = (time.time() - start) * 1000

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
        top_predictions = [
            {"class": self.class_names[idx], "confidence": float(p)}
            for p, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]

        best_class = top_predictions[0]["class"]
        best_confidence = top_predictions[0]["confidence"]

        result = {
            "predicted_class": best_class,
            "confidence": best_confidence,
            "top_predictions": top_predictions,
            "class_probabilities": {c: float(probs[i]) for i, c in enumerate(self.class_names)},
            "inference_time_ms": inference_time_ms,
            "below_confidence_threshold": best_confidence < confidence_threshold,
        }

        if result["below_confidence_threshold"]:
            result["message"] = (
                "I'm not sufficiently confident about this classification. "
                "Please upload a clearer image containing the waste item."
            )
            result["recycling_recommendation"] = None
        else:
            result["recycling_recommendation"] = get_recommendation(best_class)

        return result


# Convenience module-level function, matching the exact signature requested
# in Section 27: predict_image("path/to/image.jpg")
_predictor_singleton = None


def predict_image(path: str) -> dict:
    """
    Module-level convenience wrapper. Lazily loads (and caches) the best
    model checkpoint on first call so repeated calls don't reload weights.
    """
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = GarbagePredictor()
    return _predictor_singleton.predict_image(path)
