"""Data loading utilities with dataset-specific pre-processing."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class MissingValuePolicy:
    """Configuration describing how missing values should be handled."""

    forward_fill: bool = True
    back_fill: bool = True
    indicator: bool = True


@dataclass
class DatasetSplit:
    """Train/val/test split expressed as either ratios or explicit lengths."""

    train: float
    val: float
    test: float

    def as_lengths(self, total: int) -> Dict[str, int]:
        if self.train + self.val + self.test <= 0:
            raise ValueError("Split ratios must sum to a positive value.")
        norm = self.train + self.val + self.test
        ratios = np.array([self.train, self.val, self.test], dtype=float) / norm
        lengths = np.floor(ratios * total).astype(int)
        remainder = total - lengths.sum()
        for i in range(remainder):
            lengths[i % 3] += 1
        return {"train": int(lengths[0]), "val": int(lengths[1]), "test": int(lengths[2])}


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset with reproducible preprocessing pipeline."""

    def __init__(
        self,
        data_path: str,
        mode: str,
        seq_len: int,
        pred_len: int,
        dataset_type: str,
        split: DatasetSplit,
        stride: int = 1,
        detrend_window: Optional[int] = None,
        noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        missing_policy: Optional[MissingValuePolicy] = None,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dataset_type = dataset_type
        self.split = split
        self.stride = max(1, stride)
        self.detrend_window = detrend_window
        self.noise_fn = noise_fn
        self.missing_policy = missing_policy or MissingValuePolicy()
        self.return_dict = return_dict

        self.scaler = StandardScaler()
        self._load_and_preprocess()

    # ------------------------------------------------------------------
    def _load_and_preprocess(self) -> None:
        frame = pd.read_csv(self.data_path)
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)

        self.raw_mask = ~np.isnan(values)
        values = self._handle_missing(values)

        if self.detrend_window is not None and self.detrend_window > 1:
            trend = self._moving_average(values, window=self.detrend_window)
            values = values - trend

        total_len = values.shape[0]
        lengths = self.split.as_lengths(total_len)
        train_len = lengths["train"]
        val_len = lengths["val"]

        scaler_path = Path(self.data_path).with_suffix(".scaler.npz")
        lock_path = scaler_path.with_suffix(".lock")
        if not scaler_path.exists():
            self._fit_scaler_with_lock(values[:train_len], scaler_path, lock_path)
        self._load_scaler_params(scaler_path)

        if self.mode == "train":
            data = values[:train_len]
            mask = self.raw_mask[:train_len]
        elif self.mode == "val":
            data = values[train_len : train_len + val_len]
            mask = self.raw_mask[train_len : train_len + val_len]
        elif self.mode == "test":
            data = values[train_len + val_len :]
            mask = self.raw_mask[train_len + val_len :]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.data = self.scaler.transform(data)
        self.mask = mask.astype(np.float32)
        self._init_indices()

    # ------------------------------------------------------------------
    def _handle_missing(self, data: np.ndarray) -> np.ndarray:
        if not np.isnan(data).any():
            return data

        df = pd.DataFrame(data)
        if self.missing_policy.forward_fill:
            df = df.fillna(method="ffill")
        if self.missing_policy.back_fill:
            df = df.fillna(method="bfill")
        return df.to_numpy(dtype=np.float32)

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        window = min(window, data.shape[0])
        pad = window // 2
        padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        trend = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded)
        return trend.astype(np.float32)

    def _fit_scaler_with_lock(self, train_data: np.ndarray, scaler_path: Path, lock_path: Path) -> None:
        while True:
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    if not scaler_path.exists():
                        self.scaler.fit(train_data)
                        np.savez(scaler_path, mean=self.scaler.mean_, scale=self.scaler.scale_)
                finally:
                    os.close(lock_fd)
                    Path(lock_path).unlink(missing_ok=True)
                break
            except FileExistsError:
                time.sleep(random.uniform(0.25, 0.75))
            except Exception:
                if "lock_fd" in locals():
                    os.close(lock_fd)
                    Path(lock_path).unlink(missing_ok=True)
                raise

    def _load_scaler_params(self, scaler_path: Path) -> None:
        params = np.load(scaler_path)
        self.scaler.mean_ = params["mean"]
        self.scaler.scale_ = params["scale"]

    def _init_indices(self) -> None:
        total = self.data.shape[0]
        max_start = total - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError(
                f"Sequence length {self.seq_len} and pred_len {self.pred_len} exceed available points {total}."
            )
        self.indices = np.arange(0, max_start + 1, self.stride, dtype=int)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:  # type: ignore[override]
        idx = self.indices[index]
        input_slice = slice(idx, idx + self.seq_len)
        target_slice = slice(idx + self.seq_len, idx + self.seq_len + self.pred_len)

        inputs = self.data[input_slice].copy()
        targets = self.data[target_slice].copy()
        input_mask = self.mask[input_slice]
        target_mask = self.mask[target_slice]

        if self.noise_fn is not None and self.mode == "test":
            noisy = self.noise_fn(inputs)
            noise_mask = ~np.isnan(noisy)
            inputs = np.nan_to_num(noisy, nan=0.0)
            if self.missing_policy.indicator:
                input_mask = input_mask * noise_mask.astype(np.float32)

        sample = {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),
            "targets": torch.from_numpy(targets.astype(np.float32)),
            "input_mask": torch.from_numpy(input_mask.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
        }

        return sample if self.return_dict else (sample["inputs"], sample["targets"])


def get_dataloader(config: Dict[str, Any], mode: str = "train") -> DataLoader:
    dataset_name = config["dataset"]
    dataset_type = config.get("dataset_type", dataset_name.split("_")[0])
    split_cfg = config.get("split", {"train": 0.6, "val": 0.2, "test": 0.2})
    split = DatasetSplit(
        train=float(split_cfg.get("train", 0.6)),
        val=float(split_cfg.get("val", 0.2)),
        test=float(split_cfg.get("test", 0.2)),
    )

    detrend_window = config.get("detrend_window") if config.get("detrend", False) else None
    missing_policy = MissingValuePolicy(
        forward_fill=config.get("missing", {}).get("forward_fill", True),
        back_fill=config.get("missing", {}).get("back_fill", True),
        indicator=config.get("missing", {}).get("indicator", True),
    )

    dataset = TimeSeriesDataset(
        data_path=os.path.join(config["data_path"], f"{dataset_name}.csv"),
        mode=mode,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        dataset_type=dataset_type,
        split=split,
        stride=config.get("stride", 1),
        detrend_window=detrend_window,
        noise_fn=config.get("noise_fn"),
        missing_policy=missing_policy,
        return_dict=config.get("return_dict", True),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=(mode == "train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        drop_last=False,
    )
    return loader


ETTDataset = TimeSeriesDataset
