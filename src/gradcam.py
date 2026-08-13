"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) for the best-performing
model, so predictions can be visually explained (Section 30).

IMPORTANT: Grad-CAM is an *approximate* visual explanation of which spatial
regions most influenced the predicted class — it is not proof of the model's
internal reasoning process. Treat it as a diagnostic aid, not ground truth.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from src import config


def _find_last_conv_layer(model, model_name: str):
    """Return the last convolutional layer to hook, per architecture."""
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name in ("efficientnet_b3", "mobilenet_v3"):
        return model.features[-1]
    raise ValueError(f"Grad-CAM target layer not defined for '{model_name}'.")


class GradCAM:
    def __init__(self, model, model_name: str, device=config.DEVICE):
        self.model = model.to(device).eval()
        self.device = device
        self.target_layer = _find_last_conv_layer(model, model_name)

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: preprocessed image tensor, shape (1, 3, H, W).
        Returns a (H, W) heatmap normalised to [0, 1].
        """
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global-average-pool the gradients to get per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def overlay_heatmap(original_image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    original_image_np: HxWx3 RGB uint8 array.
    cam: 2D heatmap in [0, 1] (any resolution — will be resized).
    Returns an HxWx3 RGB uint8 array with the heatmap overlaid.
    """
    h, w = original_image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (heatmap.astype(float) * alpha + original_image_np.astype(float) * (1 - alpha))
    return np.uint8(overlay)
