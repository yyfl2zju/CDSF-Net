from collections import defaultdict

import torch


class ForgettingAwareClassBalancer:
    def __init__(
        self,
        num_classes: int,
        class_scale: float = 1.5,
        sample_scale: float = 0.2,
        max_sample_boost: float = 3.0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_scale = float(class_scale)
        self.sample_scale = float(sample_scale)
        self.max_sample_boost = float(max_sample_boost)

        self.prev_correct = {}
        self.sample_forget_count = defaultdict(int)
        self.class_seen = torch.zeros(self.num_classes, dtype=torch.float32)
        self.class_forgot = torch.zeros(self.num_classes, dtype=torch.float32)

    def _class_ratio(self) -> torch.Tensor:
        return self.class_forgot / (self.class_seen + 1.0)

    def compute_weights(self, sample_ids, labels: torch.Tensor) -> torch.Tensor:
        ratios = self._class_ratio()
        labels_cpu = labels.detach().cpu().tolist()

        weights = []
        for sample_id, label in zip(sample_ids, labels_cpu):
            label = int(label)
            class_weight = 1.0 + self.class_scale * float(ratios[label].item())
            sample_weight = 1.0 + self.sample_scale * min(
                float(self.sample_forget_count[sample_id]),
                self.max_sample_boost,
            )
            weights.append(class_weight * sample_weight)
        return labels.new_tensor(weights, dtype=torch.float32)

    @torch.no_grad()
    def update(self, sample_ids, labels: torch.Tensor, preds: torch.Tensor) -> None:
        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()

        for sample_id, label, pred in zip(sample_ids, labels_cpu, preds_cpu):
            label = int(label)
            cur_correct = int(pred) == label

            self.class_seen[label] += 1.0
            prev = self.prev_correct.get(sample_id)
            if prev is True and not cur_correct:
                self.sample_forget_count[sample_id] += 1
                self.class_forgot[label] += 1.0

            self.prev_correct[sample_id] = cur_correct
