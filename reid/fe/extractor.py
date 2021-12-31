import logging
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

sys.path.append('reid/fe/reid')
from torchreid import models


class FeatureExtractor(object):
    def __init__(self, model_type, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        self.model = models.build_model(name=model_type, num_classes=1000)
        self.model.to(self.device)
        self.model.eval()

        self.size = (128, 256)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        logging \
            .getLogger("root.tracker") \
            .info(f"\tREID MODEL :: {model_type}")

    def __call__(self, crops):
        batch = torch.cat(
            [
                self.norm(
                    cv2.resize(crop.astype(np.float32) / 255., self.size)
                ).unsqueeze(0)
                for crop in crops
            ],
            dim=0) \
            .float()
        with torch.no_grad():
            batch = batch.to(self.device)
            features = self.model(batch)
        return features.cpu().numpy()
