# reid_helper.py
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import os

class ReExtractor:
    def __init__(self, device=None, model_name="efficientnet_b0", pretrained=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Cargar EfficientNet-B0 de torchvision y quitar la última capa classifier
        self.model = models.efficientnet_b0(pretrained=pretrained)
        # Replace classifier with identity to get embeddings before final classifier
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        # Transformaciones (imagen -> tensor)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def img_to_pil(self, img_bgr):
        # img_bgr: numpy array BGR (cv2)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def get_embedding(self, crop_bgr):
        """
        crop_bgr: crop como numpy array en formato BGR (cv2)
        devuelve: embedding numpy 1D normalizado (L2 norm)
        """
        pil = self.img_to_pil(crop_bgr)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)  # shape (1, D)
            feat = feat.cpu().numpy().reshape(-1)
        # Normalizar L2
        norm = np.linalg.norm(feat) + 1e-6
        feat = feat / norm
        return feat

def cosine_similarity(a, b):
    """
    a: (D,) numpy, b: (N, D) or (D,)
    devuelve: similitudes
    """
    a = a.astype(np.float32)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    # dot products (N,)
    dots = np.dot(b, a)
    return dots  # as embeddings are L2-normalized -> dot == cosine

def save_gallery(gallery, path):
    """
    gallery: dict {id: embedding (numpy)}
    guarda un npz
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in gallery.items()})

def load_gallery(path):
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    gallery = {int(k): data[k] for k in data.files}
    return gallery
