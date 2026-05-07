import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets


DYNASTY_ORDER = {
    "两晋十六国": 0,
    "北魏": 1,
    "西魏": 2,
    "北齐": 3,
    "北周": 4,
    "隋": 5,
    "唐代": 6,
    "五代": 7,
    "宋": 8,
    "西夏": 9,
    "元": 10,
}

DEFAULT_EXCLUDED_CLASSES = {"西夏"}


def _normalize_exclude_classes(
    exclude_classes: Optional[Union[Sequence[str], str]] = None,
) -> List[str]:
    if exclude_classes is None:
        return []
    if isinstance(exclude_classes, str):
        parts = [part.strip() for part in exclude_classes.split(",")]
        return [part for part in parts if part]
    return [str(item).strip() for item in exclude_classes if str(item).strip()]


def build_final_class_names(
    exclude_classes: Optional[Union[Sequence[str], str]] = None,
) -> List[str]:
    excludes = set(DEFAULT_EXCLUDED_CLASSES)
    excludes.update(_normalize_exclude_classes(exclude_classes))
    names = sorted(DYNASTY_ORDER.keys(), key=lambda name: DYNASTY_ORDER[name])
    return [name for name in names if name not in excludes]


class ImageSamplesDataset(Dataset):
    def __init__(self, samples, transform=None, return_flag: bool = False) -> None:
        self.samples = samples
        self.transform = transform
        self.return_flag = return_flag

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        if len(item) == 3:
            path, target, is_pseudo = item
        else:
            path, target = item
            is_pseudo = 0

        img = Image.open(path).convert("RGB")
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        if self.transform is not None:
            img = self.transform(img)

        if self.return_flag:
            return img, int(target), int(is_pseudo), str(path)
        return img, int(target)


def gather_pseudo_samples(pseudo_root: str, class_to_idx: Dict[str, int]):
    pseudo_samples = []
    if not os.path.isdir(pseudo_root):
        return pseudo_samples

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = os.path.join(pseudo_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [
            os.path.join(cls_dir, name)
            for name in os.listdir(cls_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for path in files:
            pseudo_samples.append((path, cls_idx, 1))
    return pseudo_samples


def build_train_samples(clean_train_samples, class_names, pseudo_root: str, use_pseudo: bool):
    clean = [(path, label, 0) for path, label in clean_train_samples]
    pseudo = []
    if use_pseudo:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        pseudo_raw = gather_pseudo_samples(pseudo_root, class_to_idx)
        clean_abs = {os.path.abspath(path) for path, _, _ in clean}
        pseudo = [sample for sample in pseudo_raw if os.path.abspath(sample[0]) not in clean_abs]
    return clean + pseudo, clean, pseudo


def build_sample_weights(
    samples_with_flag,
    class_names,
    pseudo_weight: float = 0.4,
    small_boost: float = 2.0,
):
    clean_targets = [label for _, label, flag in samples_with_flag if flag == 0]
    class_counts = Counter(clean_targets)
    small_set = {"宋", "元"}

    class_base = {}
    for idx, name in enumerate(class_names):
        base = 1.0 / np.sqrt(class_counts.get(idx, 1))
        if name in small_set:
            base *= small_boost
        class_base[idx] = base

    weights = []
    for _, label, is_pseudo in samples_with_flag:
        weight = class_base[int(label)]
        if is_pseudo == 1:
            weight *= pseudo_weight
        weights.append(weight)
    return np.array(weights, dtype=np.float32), class_base


def load_all_samples(
    data_dir: str,
    exclude_classes: Optional[Union[Sequence[str], str]] = None,
):
    ds_raw = datasets.ImageFolder(data_dir)
    final_class_names = build_final_class_names(exclude_classes=exclude_classes)
    class_to_idx = {name: idx for idx, name in enumerate(final_class_names)}

    all_samples = []
    for path, target in ds_raw.samples:
        raw_name = ds_raw.classes[target]
        if raw_name in class_to_idx:
            all_samples.append((path, class_to_idx[raw_name]))
    return all_samples, final_class_names


def split_dataset_three_way(
    all_samples,
    val_size: float = 0.10,
    test_size: float = 0.15,
    random_state: int = 42,
):
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1.0:
        raise ValueError("Require val_size > 0, test_size > 0, and val_size + test_size < 1.0")

    targets = [sample[1] for sample in all_samples]
    indices = np.arange(len(all_samples))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )

    train_val_targets = [targets[idx] for idx in train_val_idx]
    val_ratio_in_train_val = val_size / (1.0 - test_size)
    train_idx_rel, val_idx_rel = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=val_ratio_in_train_val,
        stratify=train_val_targets,
        random_state=random_state,
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    train_samples = [all_samples[idx] for idx in train_idx]
    val_samples = [all_samples[idx] for idx in val_idx]
    test_samples = [all_samples[idx] for idx in test_idx]
    return train_samples, val_samples, test_samples


def _to_relpath(path: str, base_dir: Path) -> str:
    return os.path.relpath(os.path.abspath(path), base_dir)


def _to_abspath(path: str, base_dir: Path) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(base_dir / path)


def save_split_file(
    split_path: str,
    train_samples,
    val_samples,
    test_samples,
    class_names,
    project_root: str,
    meta=None,
) -> None:
    split_path_obj = Path(split_path)
    split_path_obj.parent.mkdir(parents=True, exist_ok=True)
    project_root_path = Path(project_root).resolve()

    payload = {
        "class_names": class_names,
        "meta": meta or {},
        "train_samples": [
            {"path": _to_relpath(path, project_root_path), "label": int(label)}
            for path, label in train_samples
        ],
        "val_samples": [
            {"path": _to_relpath(path, project_root_path), "label": int(label)}
            for path, label in val_samples
        ],
        "test_samples": [
            {"path": _to_relpath(path, project_root_path), "label": int(label)}
            for path, label in test_samples
        ],
    }
    with open(split_path_obj, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_split_file(split_path: str, project_root: str, check_exists: bool = True):
    project_root_path = Path(project_root).resolve()
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    class_names = payload["class_names"]
    train_samples = [(_to_abspath(item["path"], project_root_path), int(item["label"])) for item in payload["train_samples"]]
    val_samples = [(_to_abspath(item["path"], project_root_path), int(item["label"])) for item in payload["val_samples"]]
    test_samples = [(_to_abspath(item["path"], project_root_path), int(item["label"])) for item in payload["test_samples"]]

    if check_exists:
        for split_name, samples in (
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ):
            missing = [path for path, _ in samples if not os.path.exists(path)]
            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"Split file contains missing {split_name} paths ({len(missing)}). "
                    f"Examples:\n{preview}"
                )

    return train_samples, val_samples, test_samples, class_names, payload.get("meta", {})


def load_or_create_split_three_way(
    data_dir: str,
    split_path: str,
    project_root: str,
    random_state: int = 42,
    val_size: float = 0.10,
    test_size: float = 0.15,
    exclude_classes: Optional[Union[Sequence[str], str]] = None,
):
    exclude_list = sorted(
        list(set(DEFAULT_EXCLUDED_CLASSES).union(_normalize_exclude_classes(exclude_classes)))
    )
    exclude_set = set(exclude_list)
    target_class_names = build_final_class_names(exclude_classes=exclude_list)

    if os.path.exists(split_path):
        train_samples, val_samples, test_samples, class_names, meta = load_split_file(
            split_path=split_path,
            project_root=project_root,
            check_exists=True,
        )
        split_excludes = set(_normalize_exclude_classes(meta.get("exclude_classes", [])))
        if split_excludes == exclude_set and class_names == target_class_names:
            return train_samples, val_samples, test_samples, class_names

    all_samples, class_names = load_all_samples(data_dir, exclude_classes=exclude_list)
    train_samples, val_samples, test_samples = split_dataset_three_way(
        all_samples=all_samples,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    meta = {
        "seed": int(random_state),
        "val_size": float(val_size),
        "test_size": float(test_size),
        "train_size": float(1.0 - val_size - test_size),
        "total_samples": int(len(all_samples)),
        "exclude_classes": exclude_list,
    }
    save_split_file(
        split_path=split_path,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        class_names=class_names,
        project_root=project_root,
        meta=meta,
    )
    return train_samples, val_samples, test_samples, class_names
