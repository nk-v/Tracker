import numpy as np
import torch

from .fe.extractor import FeatureExtractor
from .rt.detection import Detection
from .rt.nn_matching import NearestNeighborDistanceMetric
from .rt.tracker import Tracker


class ReID(object):
    def __init__(self, model_type):
        self.feature_extractor = FeatureExtractor(model_type, use_cuda=True)

        self.height = self.width = None
        self.min_confidence = .5

        metric = NearestNeighborDistanceMetric("cosine", .25, 100)
        self.tracker = Tracker(metric, max_iou_distance=.7, max_age=100, n_init=3)

    def forward(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=True):
        self.height, self.width = ori_img.shape[:2]

        features = self.extract_features(bbox_xywh, ori_img)

        bbox_tlwh = self.xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        self.tracker.predict()
        self.tracker.update(detections, classes)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self.tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self.tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def extract_features(self, bbox_xywh, ori_img):
        crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self.xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            crops.append(im)
        if crops:
            features = self.feature_extractor(crops)
        else:
            features = np.array([])
        return features

    def xywh_to_tlwh(self, bbox_xywh):
        bbox_tlwh = None
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()
