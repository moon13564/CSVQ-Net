from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import hashlib
import random

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data"
DESC_ROOT = ROOT / "pre_descriptions"

DATASET_NAME = "AIGCIQA2023"
DATASET_ROOT = DATA_ROOT / DATASET_NAME

IMAGE_ROOT = DATASET_ROOT / "images"
PROMPT_FILE = DATASET_ROOT / "AIGCIQA2023_Prompts.xlsx"
INDEX_FILE = DATASET_ROOT / "pic-index.xlsx"
MOS_ROOT = DATASET_ROOT / "MOS"

DESCRIPTION_FILE = DESC_ROOT / "AIGCIQA2023_descriptions.csv"

MED_CONFIG = ROOT / "models" / "med_config.json"
BLIP_CKPT = ROOT / "models" / "model_base_retrieval_coco.pth"
OUTPUT_DIR = ROOT / "outputs" / DATASET_NAME


BLIP_MEAN = (
    0.48145466,
    0.4578275,
    0.40821073,
)

BLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711,
)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SharedTransform:
    def __init__(
        self,
        size: int = 224,
        train: bool = True,
    ):
        self.size = size
        self.train_mode = train

        self.blip_norm = transforms.Normalize(
            BLIP_MEAN,
            BLIP_STD,
        )

        self.visual_norm = transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

    def __call__(self, image: Image.Image):

        image = TF.resize(
            image,
            [256, 256],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        if self.train_mode:

            i, j, h, w = transforms.RandomCrop.get_params(
                image,
                output_size=(
                    self.size,
                    self.size,
                ),
            )

            image = TF.crop(
                image,
                i,
                j,
                h,
                w,
            )

            if random.random() < 0.5:
                image = TF.hflip(image)

        else:

            image = TF.center_crop(
                image,
                [
                    self.size,
                    self.size,
                ],
            )

        image_tensor = TF.to_tensor(image)

        semantic_image = self.blip_norm(
            image_tensor.clone()
        )

        visual_image = self.visual_norm(
            image_tensor
        )

        return semantic_image, visual_image


def load_description_map(
    csv_path: Path,
):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Description CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "image_name",
        "description",
    }

    if not required_columns.issubset(df.columns):

        raise KeyError(
            f"{csv_path.name} must contain columns "
            f"{sorted(required_columns)}.\n"
            f"Actual columns: {list(df.columns)}"
        )

    df["image_name"] = (
        df["image_name"]
        .astype(str)
        .str.strip()
    )

    df["description"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    duplicated = df[
        "image_name"
    ].duplicated()

    if duplicated.any():

        examples = (
            df.loc[
                duplicated,
                "image_name",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate image_name found "
            f"in {csv_path.name}:\n"
            f"{examples}"
        )

    description_map = dict(
        zip(
            df["image_name"],
            df["description"],
        )
    )

    return description_map


def find_image(
    image_root: Path,
    image_name: str,
):
    image_name = str(
        image_name
    ).strip()

    direct_path = (
        image_root
        / image_name
    )

    if direct_path.is_file():
        return direct_path

    matches = list(
        image_root.rglob(
            image_name
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple image files named "
            f"{image_name} found under:\n"
            f"{image_root}"
        )

    raise FileNotFoundError(
        f"Image not found:\n"
        f"{image_name}\n"
        f"under:\n"
        f"{image_root}"
    )


def _first_numeric_array_from_mat(
    mat_dict,
    preferred_key: str,
):
    if preferred_key in mat_dict:

        arr = np.asarray(
            mat_dict[
                preferred_key
            ]
        ).squeeze()

        if np.issubdtype(
            arr.dtype,
            np.number,
        ):
            return arr.astype(
                np.float64
            )

    candidates = []

    for key, value in mat_dict.items():

        if key.startswith("__"):
            continue

        arr = np.asarray(
            value
        ).squeeze()

        if (
            arr.size > 1
            and np.issubdtype(
                arr.dtype,
                np.number,
            )
        ):
            candidates.append(
                (
                    key,
                    arr,
                )
            )

    if not candidates:

        raise RuntimeError(
            "No numeric array found "
            "inside the MAT file.\n"
            f"Available keys: "
            f"{list(mat_dict.keys())}"
        )

    for _, arr in candidates:

        if arr.size == 2400:
            return arr.astype(
                np.float64
            )

    return candidates[
        0
    ][1].astype(
        np.float64
    )


def load_mos_vector(
    mos_root: Path,
    key: str = "mosz1",
):
    if not mos_root.is_dir():

        raise FileNotFoundError(
            f"MOS folder not found:\n"
            f"{mos_root}"
        )

    matches = []

    for p in mos_root.rglob(
        "*"
    ):

        if (
            p.is_file()
            and key.lower()
            in p.stem.lower()
        ):
            matches.append(
                p
            )

    if not matches:

        raise FileNotFoundError(
            f"Cannot find a MOS file "
            f"whose filename contains '{key}' "
            f"under:\n"
            f"{mos_root}"
        )

    priority = {
        ".mat": 0,
        ".npy": 1,
        ".txt": 2,
        ".csv": 3,
        ".xlsx": 4,
        ".xls": 5,
    }

    matches.sort(
        key=lambda p:
        priority.get(
            p.suffix.lower(),
            99,
        )
    )

    path = matches[0]

    suffix = (
        path
        .suffix
        .lower()
    )

    if suffix == ".mat":

        try:
            from scipy.io import loadmat

        except ImportError as e:

            raise ImportError(
                "Reading .mat MOS files requires scipy.\n"
                "Install with:\n"
                "pip install scipy"
            ) from e

        mat_data = loadmat(
            path
        )

        mos = _first_numeric_array_from_mat(
            mat_data,
            key,
        )

    elif suffix == ".npy":

        mos = np.load(
            path
        ).squeeze()

    elif suffix == ".txt":

        mos = np.loadtxt(
            path
        ).squeeze()

    elif suffix == ".csv":

        raw = pd.read_csv(
            path,
            header=None,
        )

        mos = pd.to_numeric(
            raw.stack(),
            errors="coerce",
        ).dropna().to_numpy()

    elif suffix in {
        ".xlsx",
        ".xls",
    }:

        raw = pd.read_excel(
            path,
            header=None,
        )

        mos = pd.to_numeric(
            raw.stack(),
            errors="coerce",
        ).dropna().to_numpy()

    else:

        raise RuntimeError(
            f"Unsupported MOS file:\n"
            f"{path}"
        )

    mos = np.asarray(
        mos,
        dtype=np.float64,
    ).reshape(-1)

    print(
        f"[AIGCIQA2023] "
        f"Quality MOS file: "
        f"{path.name}"
    )

    print(
        f"[AIGCIQA2023] "
        f"MOS count: "
        f"{len(mos)}"
    )

    return mos


def read_metadata():
    if not PROMPT_FILE.is_file():

        raise FileNotFoundError(
            f"Prompt file not found:\n"
            f"{PROMPT_FILE}"
        )

    if not INDEX_FILE.is_file():

        raise FileNotFoundError(
            f"pic-index file not found:\n"
            f"{INDEX_FILE}"
        )

    prompts = pd.read_excel(
        PROMPT_FILE,
        header=None,
    )

    if prompts.shape[1] < 3:

        raise RuntimeError(
            "Unexpected "
            "AIGCIQA2023_Prompts.xlsx "
            f"shape: {prompts.shape}"
        )

    prompt_list = (
        prompts
        .iloc[:, 2]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(
        "[AIGCIQA2023] "
        "prompt count:",
        len(prompt_list),
    )

    if len(prompt_list) != 100:

        raise RuntimeError(
            "Expected 100 "
            "AIGCIQA2023 prompts, "
            f"got {len(prompt_list)}"
        )

    print(
        "[AIGCIQA2023] "
        "first prompt:",
        prompt_list[0],
    )

    index_df = pd.read_excel(
        INDEX_FILE,
        header=None,
    )

    if index_df.shape[1] < 2:

        raise RuntimeError(
            f"Unexpected pic-index "
            f"shape: {index_df.shape}"
        )

    index_df = (
        index_df
        .iloc[:, :2]
        .copy()
    )

    index_df.columns = [
        "model",
        "original_image_name",
    ]

    if len(index_df) != 2400:

        raise RuntimeError(
            "Expected 2400 rows "
            "in pic-index.xlsx, "
            f"got {len(index_df)}"
        )

    mos = load_mos_vector(
        MOS_ROOT,
        key="mosz1",
    )

    if len(mos) != 2400:

        raise RuntimeError(
            "Expected 2400 "
            "quality MOS values, "
            f"got {len(mos)}"
        )

    descriptions = load_description_map(
        DESCRIPTION_FILE
    )

    if len(descriptions) != 2400:

        raise RuntimeError(
            "Expected 2400 "
            "descriptive prompts, "
            f"got {len(descriptions)}"
        )

    expected_names = {
        f"{i}.png"
        for i in range(
            2400
        )
    }

    actual_names = set(
        descriptions.keys()
    )

    if expected_names != actual_names:

        missing = sorted(
            expected_names
            - actual_names
        )[:10]

        extra = sorted(
            actual_names
            - expected_names
        )[:10]

        raise RuntimeError(
            "AIGCIQA2023 description image names "
            "are not exactly 0.png ... 2399.png.\n"
            f"Missing examples: {missing}\n"
            f"Extra examples: {extra}"
        )

    rows = []

    for i, row in index_df.iterrows():

        original_image_name = (
            str(
                row[
                    "original_image_name"
                ]
            )
            .strip()
        )

        generator_model = (
            str(
                row[
                    "model"
                ]
            )
            .strip()
        )

        stem = Path(
            original_image_name
        ).stem

        prefix = stem.split(
            "-"
        )[0]

        try:
            prompt_index = int(
                prefix
            )

        except ValueError as e:

            raise RuntimeError(
                "Cannot derive "
                "prompt index from "
                f"official image name:\n"
                f"{original_image_name}"
            ) from e

        if not (
            0
            <= prompt_index
            < len(prompt_list)
        ):

            raise RuntimeError(
                "Prompt index "
                f"{prompt_index} "
                "out of range for:\n"
                f"{original_image_name}"
            )

        local_image_name = (
            f"{i}.png"
        )

        rows.append(
            {
                "image_name":
                    local_image_name,

                "original_image_name":
                    original_image_name,

                "model":
                    generator_model,

                "prompt":
                    prompt_list[
                        prompt_index
                    ],

                "mos":
                    float(
                        mos[i]
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(df) != 2400:

        raise RuntimeError(
            f"Final metadata "
            f"contains {len(df)} "
            "rows instead of 2400"
        )

    return df


def validate_metadata(
    df: pd.DataFrame,
    image_root: Path,
    description_map,
):
    if len(df) == 0:

        raise RuntimeError(
            "AIGCIQA2023 metadata is empty"
        )

    missing_descriptions = [
        name
        for name
        in df[
            "image_name"
        ].astype(str)
        if name
        not in description_map
    ]

    if missing_descriptions:

        raise RuntimeError(
            f"{len(missing_descriptions)} "
            "images do not have "
            "descriptive prompts.\n"
            f"Examples:\n"
            f"{missing_descriptions[:10]}"
        )

    missing_images = []

    for name in df[
        "image_name"
    ].astype(str):

        try:
            find_image(
                image_root,
                name,
            )

        except FileNotFoundError:

            missing_images.append(
                name
            )

            if len(
                missing_images
            ) >= 10:
                break

    if missing_images:

        raise RuntimeError(
            "Image files are missing.\n"
            f"Examples:\n"
            f"{missing_images}"
        )

    print(
        f"[AIGCIQA2023] "
        f"metadata={len(df)}, "
        f"descriptions="
        f"{len(description_map)}, "
        f"pairing check=OK"
    )


class AIGCIQA2023Dataset(
    Dataset
):

    def __init__(
        self,
        train: bool = True,
        dataframe=None,
    ):

        if dataframe is None:
            self.df = (
                read_metadata()
            )

        else:
            self.df = (
                dataframe
                .reset_index(
                    drop=True
                )
            )

        self.description_map = (
            load_description_map(
                DESCRIPTION_FILE
            )
        )

        self.transform = (
            SharedTransform(
                size=224,
                train=train,
            )
        )

        validate_metadata(
            self.df,
            IMAGE_ROOT,
            self.description_map,
        )

    def clone_for_split(
        self,
        train: bool,
    ):
        obj = object.__new__(
            AIGCIQA2023Dataset
        )

        obj.df = (
            self.df
            .reset_index(
                drop=True
            )
        )

        obj.description_map = (
            self.description_map
        )

        obj.transform = (
            SharedTransform(
                size=224,
                train=train,
            )
        )

        return obj

    def __len__(
        self
    ):
        return len(
            self.df
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]

        image_name = str(
            row[
                "image_name"
            ]
        )

        image_path = find_image(
            IMAGE_ROOT,
            image_name,
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        semantic_image, visual_image = (
            self.transform(
                image
            )
        )

        description = (
            self.description_map[
                image_name
            ]
        )

        mos = torch.tensor(
            float(
                row[
                    "mos"
                ]
            ),
            dtype=torch.float32,
        )

        return {
            "semantic_image":
                semantic_image,

            "visual_image":
                visual_image,

            "original_prompt":
                str(
                    row[
                        "prompt"
                    ]
                ),

            "descriptive_prompt":
                description,

            "mos":
                mos,

            "image_name":
                image_name,

            "original_image_name":
                str(
                    row[
                        "original_image_name"
                    ]
                ),

            "generator_model":
                str(
                    row[
                        "model"
                    ]
                ),
        }

def rankdata_numpy(x):
    return pd.Series(np.asarray(x)).rank(method="average").to_numpy()


def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def logistic_5param(x, beta1, beta2, beta3, beta4, beta5):
    x = np.asarray(x, dtype=np.float64)
    z = np.clip(beta2 * (x - beta3), -60.0, 60.0)
    return beta1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + beta4 * x + beta5


def fit_5param_logistic(pred, gt):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)

    if len(pred) < 6 or np.std(pred) < 1e-12 or np.std(gt) < 1e-12:
        return None, None, False

    pred_std = max(float(np.std(pred)), 1e-6)
    gt_range = max(float(np.max(gt) - np.min(gt)), 1e-6)

    p0 = np.asarray([
        -gt_range,
        1.0 / pred_std,
        float(np.mean(pred)),
        0.0,
        float(np.mean(gt)),
    ], dtype=np.float64)

    try:
        params, _ = curve_fit(
            logistic_5param,
            pred,
            gt,
            p0=p0,
            maxfev=50000,
        )
        mapped = logistic_5param(pred, *params)
        if not np.all(np.isfinite(mapped)):
            return None, None, False
        return mapped, params.astype(np.float64), True
    except Exception:
        return None, None, False


def compute_metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if len(pred) < 2:
        return {
            "SRCC": float("nan"),
            "PLCC_raw": float("nan"),
            "RMSE_raw": float("nan"),
            "MainScore_raw": float("nan"),
            "PLCC_5PL": float("nan"),
            "RMSE_5PL": float("nan"),
            "MainScore_5PL": float("nan"),
            "Fit5PL_OK": 0,
            "Fit5PL_params": None,
        }

    srcc = _safe_corrcoef(rankdata_numpy(pred), rankdata_numpy(gt))
    plcc_raw = _safe_corrcoef(pred, gt)
    rmse_raw = float(np.sqrt(np.mean((pred - gt) ** 2)))
    main_raw = float((srcc + plcc_raw) / 2.0)

    mapped_pred, params, fit_ok = fit_5param_logistic(pred, gt)
    if fit_ok:
        plcc_5pl = _safe_corrcoef(mapped_pred, gt)
        rmse_5pl = float(np.sqrt(np.mean((mapped_pred - gt) ** 2)))
        main_5pl = float((srcc + plcc_5pl) / 2.0)
        params_list = [float(v) for v in params]
    else:
        plcc_5pl = float("nan")
        rmse_5pl = float("nan")
        main_5pl = float("nan")
        params_list = None

    return {
        "SRCC": float(srcc),
        "PLCC_raw": float(plcc_raw),
        "RMSE_raw": rmse_raw,
        "MainScore_raw": main_raw,
        "PLCC_5PL": float(plcc_5pl),
        "RMSE_5PL": rmse_5pl,
        "MainScore_5PL": main_5pl,
        "Fit5PL_OK": int(fit_ok),
        "Fit5PL_params": params_list,
    }


PROMPT_SPLIT_METHOD = "prompt_level_group_split_v1"
RANDOM_SPLIT_METHOD = "random_image_split_v1"


def split_indices_by_prompt(df, val_ratio=0.2, seed=42):
    if "prompt" not in df.columns:
        raise RuntimeError(
            "Prompt-level split requires metadata column 'prompt', "
            f"but current columns are: {list(df.columns)}"
        )

    if "image_name" not in df.columns:
        raise RuntimeError(
            "Prompt-level split requires metadata column 'image_name', "
            f"but current columns are: {list(df.columns)}"
        )

    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(
            f"val_ratio must be between 0 and 1, got {val_ratio}"
        )

    raw_prompts = df["prompt"]
    prompt_text = raw_prompts.astype(str).str.strip()

    empty_mask = raw_prompts.isna() | prompt_text.eq("")
    if empty_mask.any():
        bad_rows = df.loc[empty_mask, "image_name"].astype(str).head(10).tolist()
        raise RuntimeError(
            f"Found {int(empty_mask.sum())} samples with empty original prompts.\n"
            f"Examples: {bad_rows}"
        )

    prompts = prompt_text

    unique_prompts = sorted(pd.unique(prompts).tolist())

    if len(unique_prompts) < 2:
        raise RuntimeError(
            "Prompt-level train/validation split requires at least 2 unique prompts, "
            f"but found {len(unique_prompts)}."
        )

    prompt_counts = prompts.value_counts()
    count_min = int(prompt_counts.min())
    count_max = int(prompt_counts.max())
    count_median = float(prompt_counts.median())
    count_mean = float(prompt_counts.mean())

    shuffled_prompts = np.asarray(unique_prompts, dtype=object).copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled_prompts)

    val_prompt_num = int(round(len(unique_prompts) * float(val_ratio)))
    val_prompt_num = max(1, min(val_prompt_num, len(unique_prompts) - 1))

    val_prompt_list = shuffled_prompts[:val_prompt_num].tolist()
    train_prompt_list = shuffled_prompts[val_prompt_num:].tolist()

    train_prompt_set = set(train_prompt_list)
    val_prompt_set = set(val_prompt_list)

    train_mask = prompts.isin(train_prompt_set).to_numpy()
    val_mask = prompts.isin(val_prompt_set).to_numpy()

    train_idx = np.flatnonzero(train_mask).tolist()
    val_idx = np.flatnonzero(val_mask).tolist()

    prompt_overlap = train_prompt_set & val_prompt_set
    image_overlap = set(train_idx) & set(val_idx)

    assert len(prompt_overlap) == 0, (
        f"Prompt leakage detected: {len(prompt_overlap)} overlapping prompts."
    )
    assert len(image_overlap) == 0, (
        f"Image-index overlap detected: {len(image_overlap)} overlapping indices."
    )
    assert len(train_idx) + len(val_idx) == len(df), (
        "Some samples were not assigned exactly once: "
        f"train={len(train_idx)}, val={len(val_idx)}, total={len(df)}"
    )
    assert bool(np.all(train_mask | val_mask)), (
        "At least one sample was not assigned to train or validation."
    )
    assert not bool(np.any(train_mask & val_mask)), (
        "At least one sample was assigned to both train and validation."
    )

    signature_payload = {
        "method": PROMPT_SPLIT_METHOD,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "train_prompts": sorted(train_prompt_list),
        "val_prompts": sorted(val_prompt_list),
    }
    split_signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    split_info = {
        "method": PROMPT_SPLIT_METHOD,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "signature": split_signature,
        "total_samples": int(len(df)),
        "total_prompts": int(len(unique_prompts)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_prompts": int(len(train_prompt_list)),
        "val_prompts": int(len(val_prompt_list)),
        "prompt_overlap": int(len(prompt_overlap)),
        "train_sample_ratio": float(len(train_idx) / len(df)),
        "val_sample_ratio": float(len(val_idx) / len(df)),
        "prompt_images_min": count_min,
        "prompt_images_max": count_max,
        "prompt_images_median": count_median,
        "prompt_images_mean": count_mean,
    }

    print("\n" + "=" * 70)
    print("Prompt-Level 8:2 Group Split")
    print("=" * 70)
    print(f"Total samples          : {split_info['total_samples']}")
    print(f"Total unique prompts   : {split_info['total_prompts']}")
    print(f"Train samples          : {split_info['train_samples']}")
    print(f"Val samples            : {split_info['val_samples']}")
    print(f"Train prompts          : {split_info['train_prompts']}")
    print(f"Val prompts            : {split_info['val_prompts']}")
    print(f"Prompt overlap         : {split_info['prompt_overlap']}")
    print(f"Train sample ratio     : {split_info['train_sample_ratio']:.4f}")
    print(f"Val sample ratio       : {split_info['val_sample_ratio']:.4f}")
    print(
        "Images per prompt       : "
        f"min={count_min}, max={count_max}, "
        f"median={count_median:.2f}, mean={count_mean:.2f}"
    )
    print(f"Split seed             : {seed}")
    print(f"Split signature        : {split_signature[:16]}...")
    print("=" * 70 + "\n")

    return train_idx, val_idx, split_info


def split_indices_random_image(df, val_ratio=0.2, seed=42):
    if "image_name" not in df.columns:
        raise RuntimeError(
            "Random image split requires metadata column 'image_name', "
            f"but current columns are: {list(df.columns)}"
        )

    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    total = len(df)
    if total < 2:
        raise RuntimeError("Random image split requires at least 2 samples.")

    indices = np.arange(total, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_num = int(round(total * float(val_ratio)))
    val_num = max(1, min(val_num, total - 1))

    val_idx = indices[:val_num].tolist()
    train_idx = indices[val_num:].tolist()

    image_overlap = set(train_idx) & set(val_idx)
    assert len(image_overlap) == 0
    assert len(train_idx) + len(val_idx) == total

    prompts = df["prompt"].fillna("").astype(str).str.strip() if "prompt" in df.columns else None
    if prompts is not None:
        train_prompt_set = set(prompts.iloc[train_idx].tolist())
        val_prompt_set = set(prompts.iloc[val_idx].tolist())
        prompt_overlap = train_prompt_set & val_prompt_set
        total_prompts = int(prompts.nunique())
        train_prompts = len(train_prompt_set)
        val_prompts = len(val_prompt_set)
        prompt_overlap_count = len(prompt_overlap)
    else:
        total_prompts = 0
        train_prompts = 0
        val_prompts = 0
        prompt_overlap_count = 0

    signature_payload = {
        "method": RANDOM_SPLIT_METHOD,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "train_indices": sorted(int(i) for i in train_idx),
        "val_indices": sorted(int(i) for i in val_idx),
    }
    split_signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    split_info = {
        "method": RANDOM_SPLIT_METHOD,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "signature": split_signature,
        "total_samples": int(total),
        "total_prompts": int(total_prompts),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_prompts": int(train_prompts),
        "val_prompts": int(val_prompts),
        "prompt_overlap": int(prompt_overlap_count),
        "train_sample_ratio": float(len(train_idx) / total),
        "val_sample_ratio": float(len(val_idx) / total),
    }

    print("\n" + "=" * 70)
    print("Random Image-Level 8:2 Split")
    print("=" * 70)
    print(f"Total samples          : {split_info['total_samples']}")
    print(f"Total unique prompts   : {split_info['total_prompts']}")
    print(f"Train samples          : {split_info['train_samples']}")
    print(f"Val samples            : {split_info['val_samples']}")
    print(f"Train prompts          : {split_info['train_prompts']}")
    print(f"Val prompts            : {split_info['val_prompts']}")
    print(f"Prompt overlap         : {split_info['prompt_overlap']}")
    print(f"Train sample ratio     : {split_info['train_sample_ratio']:.4f}")
    print(f"Val sample ratio       : {split_info['val_sample_ratio']:.4f}")
    print(f"Split seed             : {seed}")
    print(f"Split signature        : {split_signature[:16]}...")
    print("=" * 70 + "\n")

    return train_idx, val_idx, split_info


def save_prompt_split_files(
    df,
    train_idx,
    val_idx,
    split_info,
    output_dir,
    dataset_name,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(split_info["seed"])
    split_csv = output_dir / f"split_seed{seed}.csv"
    summary_json = output_dir / f"split_seed{seed}_summary.json"

    split_labels = np.empty(len(df), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"

    split_df = pd.DataFrame(
        {
            "image_name": df["image_name"].astype(str).to_numpy(),
            "prompt": df["prompt"].astype(str).str.strip().to_numpy(),
            "split": split_labels,
        }
    )

    if split_csv.exists():
        old_split = pd.read_csv(split_csv)
        compare_cols = ["image_name", "prompt", "split"]
        if (
            list(old_split.columns) != compare_cols
            or len(old_split) != len(split_df)
            or not old_split[compare_cols].astype(str).equals(
                split_df[compare_cols].astype(str)
            )
        ):
            raise RuntimeError(
                f"Existing split file differs from the newly generated split:\n"
                f"{split_csv}\n"
                "Refusing to overwrite it. Please inspect the dataset/seed first."
            )
    else:
        split_df.to_csv(
            split_csv,
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "dataset_name": dataset_name,
        **split_info,
    }

    if summary_json.exists():
        old_summary = json.loads(
            summary_json.read_text(encoding="utf-8")
        )
        if old_summary.get("signature") != summary["signature"]:
            raise RuntimeError(
                f"Existing split summary has a different signature:\n"
                f"{summary_json}\n"
                "Refusing to overwrite it."
            )
    else:
        summary_json.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Split CSV saved        : {split_csv}")
    print(f"Split summary saved    : {summary_json}")

    return split_csv, summary_json


def save_random_split_files(
    df,
    train_idx,
    val_idx,
    split_info,
    output_dir,
    dataset_name,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(split_info["seed"])
    split_csv = output_dir / f"split_random_seed{seed}.csv"
    summary_json = output_dir / f"split_random_seed{seed}_summary.json"

    split_labels = np.empty(len(df), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"

    split_df = pd.DataFrame({
        "image_name": df["image_name"].astype(str).to_numpy(),
        "prompt": df["prompt"].fillna("").astype(str).str.strip().to_numpy()
        if "prompt" in df.columns else np.full(len(df), "", dtype=object),
        "split": split_labels,
    })

    if split_csv.exists():
        old_split = pd.read_csv(split_csv, keep_default_na=False)
        compare_cols = ["image_name", "prompt", "split"]
        if (
            list(old_split.columns) != compare_cols
            or len(old_split) != len(split_df)
            or not old_split[compare_cols].astype(str).equals(
                split_df[compare_cols].astype(str)
            )
        ):
            raise RuntimeError(
                f"Existing random split file differs from the newly generated split:\n"
                f"{split_csv}\nRefusing to overwrite it."
            )
    else:
        split_df.to_csv(split_csv, index=False, encoding="utf-8-sig")

    summary = {"dataset_name": dataset_name, **split_info}
    if summary_json.exists():
        old_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        if old_summary.get("signature") != summary["signature"]:
            raise RuntimeError(
                f"Existing random split summary has a different signature:\n"
                f"{summary_json}\nRefusing to overwrite it."
            )
    else:
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Split CSV saved        : {split_csv}")
    print(f"Split summary saved    : {summary_json}")
    return split_csv, summary_json


def verify_resume_split(checkpoint, current_split_info):
    checkpoint_split = checkpoint.get("split_info")

    if checkpoint_split is None:
        raise RuntimeError(
            "This checkpoint does not contain split information.\n"
            "Start a fresh run instead of resuming this checkpoint."
        )

    keys_to_check = [
        "method",
        "seed",
        "val_ratio",
        "signature",
    ]

    mismatches = []
    for key in keys_to_check:
        old_value = checkpoint_split.get(key)
        new_value = current_split_info.get(key)

        if key == "val_ratio":
            same = (
                old_value is not None
                and abs(float(old_value) - float(new_value)) < 1e-12
            )
        else:
            same = old_value == new_value

        if not same:
            mismatches.append(
                f"{key}: checkpoint={old_value!r}, current={new_value!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Resume split mismatch detected.\n"
            + "\n".join(mismatches)
            + "\nRefusing to continue training with a different split."
        )


CSV_FIELDS = [
    "epoch",
    "loss",
    "SRCC",
    "PLCC_raw",
    "RMSE_raw",
    "MainScore_raw",
    "PLCC_5PL",
    "RMSE_5PL",
    "MainScore_5PL",
    "Fit5PL_OK",
    "W_sem",
    "W_vis",
    "best_epoch",
    "best_main_score_raw",
    "no_improve_epochs",
    "lr_semantic",
    "lr_visual",
    "lr_fusion",
    "lr_moe",
]


def init_metrics_csv(csv_path: Path, resume: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and csv_path.exists():
        return

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_metrics_csv(csv_path: Path, row: dict):
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    predictions = []
    ground_truth = []

    sem_weight_sum = 0.0
    vis_weight_sum = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        sem = batch["semantic_image"].to(device, non_blocking=True)
        vis = batch["visual_image"].to(device, non_blocking=True)

        score, aux = model(
            sem,
            vis,
            batch["original_prompt"],
            batch["descriptive_prompt"],
            return_aux=True,
        )

        predictions.extend(score.detach().cpu().tolist())
        ground_truth.extend(batch["mos"].tolist())

        branch_weights = aux["fusion"]["branch_weights"]
        b = branch_weights.shape[0]

        sem_weight_sum += branch_weights[:, 0].sum().item()
        vis_weight_sum += branch_weights[:, 1].sum().item()
        total_samples += b

    metrics = compute_metrics(predictions, ground_truth)
    metrics["W_sem"] = sem_weight_sum / max(total_samples, 1)
    metrics["W_vis"] = vis_weight_sum / max(total_samples, 1)
    return metrics


def train_model(
    dataset,
    output_dir,
    epochs=30,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-2,
    num_workers=8,
    prefetch_factor=4,
    split_mode="prompt",
    val_ratio=0.2,
    seed=42,
    resume_path="",
    patience=8,
    min_delta=1e-4,
):
    from models.model import DualBranchAGIQA

    seed_everything(seed)

    if not MED_CONFIG.is_file():
        raise FileNotFoundError(f"Missing: {MED_CONFIG}")
    if not BLIP_CKPT.is_file():
        raise FileNotFoundError(f"Missing: {BLIP_CKPT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_mode = str(split_mode).strip().lower()
    if split_mode not in {"prompt", "random"}:
        raise ValueError(f"split_mode must be 'prompt' or 'random', got {split_mode!r}")

    if split_mode == "prompt":
        metrics_csv = output_dir / "metrics.csv"
        last_ckpt_path = output_dir / "last.pt"
        best_ckpt_path = output_dir / "best.pt"
        train_idx, val_idx, split_info = split_indices_by_prompt(
            dataset.df, val_ratio=val_ratio, seed=seed
        )
        save_prompt_split_files(
            dataset.df, train_idx, val_idx, split_info, output_dir, DATASET_NAME
        )
    else:
        metrics_csv = output_dir / "metrics_random.csv"
        last_ckpt_path = output_dir / "last_random.pt"
        best_ckpt_path = output_dir / "best_random.pt"
        train_idx, val_idx, split_info = split_indices_random_image(
            dataset.df, val_ratio=val_ratio, seed=seed
        )
        save_random_split_files(
            dataset.df, train_idx, val_idx, split_info, output_dir, DATASET_NAME
        )

    print(f"Split mode             : {split_mode}")
    train_ds = dataset.clone_for_split(train=True)
    val_ds = dataset.clone_for_split(train=False)

    loader_common = {
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }

    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        **loader_common,
    )

    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        **loader_common,
    )

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples  : {len(val_idx)}")

    model = DualBranchAGIQA(
        med_config_path=str(MED_CONFIG),
        blip_pretrained_path=str(BLIP_CKPT),
        image_size=224,
        embed_dim=256,
        visual_pretrained=True,
        num_experts=4,
        top_k=2,
    ).to(device)

    semantic_trainable_params = [
        p for p in model.semantic.parameters()
        if p.requires_grad
    ]

    semantic_params = [
        p for p in model.semantic.parameters()
        if p.requires_grad
    ]

    visual_params = [
        p for p in model.visual.parameters()
        if p.requires_grad
    ]

    fusion_params = [
        p for p in model.fusion.parameters()
        if p.requires_grad
    ]

    moe_params = [
        p for p in model.regressor.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": semantic_params,
                "lr": lr * 0.1,
                "name": "semantic",
            },
            {
                "params": visual_params,
                "lr": lr,
                "name": "visual",
            },
            {
                "params": fusion_params,
                "lr": lr * 0.5,
                "name": "fusion",
            },
            {
                "params": moe_params,
                "lr": lr,
                "name": "moe",
            },
        ],
        weight_decay=weight_decay,
    )

    semantic_freeze_epochs = 3

    def set_semantic_stage_trainable(enabled):
        for p in semantic_trainable_params:
            p.requires_grad = enabled

    set_semantic_stage_trainable(False)


    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_main_score = -1e9
    best_epoch = 0
    no_improve_epochs = 0

    resume_enabled = bool(resume_path)

    if resume_enabled:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path

        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print("\n" + "=" * 70)
        print(f"Resuming from: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        verify_resume_split(checkpoint, split_info)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1

        if "best_main_score" in checkpoint:
            best_main_score = float(checkpoint["best_main_score"])
            best_epoch = int(checkpoint.get("best_epoch", checkpoint_epoch))
            no_improve_epochs = int(checkpoint.get("no_improve_epochs", 0))
        elif "metrics" in checkpoint:
            old_metrics = checkpoint["metrics"]
            best_main_score = float(old_metrics.get("MainScore", -1e9))
            best_epoch = checkpoint_epoch
            no_improve_epochs = 0

        print(f"Checkpoint epoch : {checkpoint_epoch}")
        print(f"Continue from    : Epoch {start_epoch}")
        print(f"Best epoch       : {best_epoch}")
        print(f"Best MainScore_raw   : {best_main_score:.6f}")
        print(f"Early-stop count : {no_improve_epochs}/{patience}")
        print("=" * 70 + "\n")

    if start_epoch <= semantic_freeze_epochs:
        set_semantic_stage_trainable(False)
        print(
            f"Semantic stage        : FROZEN "
            f"(Epoch {start_epoch}-{semantic_freeze_epochs})"
        )
    else:
        set_semantic_stage_trainable(True)
        print(
            f"Semantic stage        : UNFROZEN "
            f"(starting from Epoch {start_epoch})"
        )

    init_metrics_csv(metrics_csv, resume=resume_enabled)

    if start_epoch > epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}, "
            f"but --epochs={epochs}. Nothing to train."
        )
        return model

    for epoch in range(start_epoch, epochs + 1):
        if epoch <= semantic_freeze_epochs:
            set_semantic_stage_trainable(False)
            semantic_stage = "FROZEN"
        else:
            set_semantic_stage_trainable(True)
            semantic_stage = "UNFROZEN"

        model.train()

        print(
            f"Semantic stage: {semantic_stage} | "
            f"Epoch {epoch}/{epochs}"
        )
        running_loss = 0.0
        sample_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            sem = batch["semantic_image"].to(device, non_blocking=True)
            vis = batch["visual_image"].to(device, non_blocking=True)
            mos = batch["mos"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(
                    sem,
                    vis,
                    batch["original_prompt"],
                    batch["descriptive_prompt"],
                )
                loss = F.smooth_l1_loss(pred, mos)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            b = mos.shape[0]
            running_loss += loss.item() * b
            sample_count += b
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_loss / max(sample_count, 1)
        metrics = evaluate(model, val_loader, device)
        current_main_score = metrics["MainScore_raw"]

        improved = current_main_score > (best_main_score + min_delta)
        if improved:
            best_main_score = current_main_score
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        current_lr_semantic = optimizer.param_groups[0]["lr"]
        current_lr_visual = optimizer.param_groups[1]["lr"]
        current_lr_fusion = optimizer.param_groups[2]["lr"]
        current_lr_moe = optimizer.param_groups[3]["lr"]

        print(
            f"\nEpoch {epoch}: "
            f"loss={train_loss:.6f}, "
            f"SRCC={metrics['SRCC']:.4f}, "
            f"PLCC_raw={metrics['PLCC_raw']:.4f}, "
            f"RMSE_raw={metrics['RMSE_raw']:.4f}"
        )
        print(
            f"5PL metrics: "
            f"PLCC_5PL={metrics['PLCC_5PL']:.4f}, "
            f"RMSE_5PL={metrics['RMSE_5PL']:.4f}, "
            f"MainScore_5PL={metrics['MainScore_5PL']:.4f}, "
            f"FitOK={metrics['Fit5PL_OK']}"
        )
        print(
            "Branch weights (validation mean): "
            f"W_sem={metrics['W_sem']:.4f}, "
            f"W_vis={metrics['W_vis']:.4f}"
        )
        print(
            f"MainScore_raw={current_main_score:.4f}, "
            f"Best={best_main_score:.4f} (Epoch {best_epoch}), "
            f"EarlyStop={no_improve_epochs}/{patience}"
        )

        print(
            f"Learning rates: "
            f"Semantic={current_lr_semantic:.2e}, "
            f"Visual={current_lr_visual:.2e}, "
            f"Fusion={current_lr_fusion:.2e}, "
            f"MoE={current_lr_moe:.2e}"
        )

        append_metrics_csv(
            metrics_csv,
            {
                "epoch": epoch,
                "loss": f"{train_loss:.8f}",
                "SRCC": f"{metrics['SRCC']:.8f}",
                "PLCC_raw": f"{metrics['PLCC_raw']:.8f}",
                "RMSE_raw": f"{metrics['RMSE_raw']:.8f}",
                "MainScore_raw": f"{current_main_score:.8f}",
                "PLCC_5PL": f"{metrics['PLCC_5PL']:.8f}",
                "RMSE_5PL": f"{metrics['RMSE_5PL']:.8f}",
                "MainScore_5PL": f"{metrics['MainScore_5PL']:.8f}",
                "Fit5PL_OK": metrics["Fit5PL_OK"],
                "W_sem": f"{metrics['W_sem']:.8f}",
                "W_vis": f"{metrics['W_vis']:.8f}",
                "best_epoch": best_epoch,
                "best_main_score_raw": f"{best_main_score:.8f}",
                "no_improve_epochs": no_improve_epochs,
                "lr_semantic": f"{current_lr_semantic:.10g}",
                "lr_visual": f"{current_lr_visual:.10g}",
                "lr_fusion": f"{current_lr_fusion:.10g}",
                "lr_moe": f"{current_lr_moe:.10g}",
            },
        )

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "metrics": metrics,
            "best_main_score": best_main_score,
            "best_main_score_raw": best_main_score,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "seed": seed,
            "val_ratio": val_ratio,
            "split_mode": split_mode,
            "split_info": split_info,
            "semantic_freeze_epochs": semantic_freeze_epochs,
            "semantic_stage": semantic_stage,
        }

        torch.save(checkpoint_state, last_ckpt_path)

        if improved:
            torch.save(checkpoint_state, best_ckpt_path)
            print("Saved new best checkpoint.")

        print(f"CSV updated: {metrics_csv}")
        print(f"Checkpoint: {last_ckpt_path}")

        if no_improve_epochs >= patience:
            print("\n" + "=" * 70)
            print("EARLY STOPPING")
            print(
                f"No sufficient MainScore improvement for "
                f"{patience} consecutive epochs."
            )
            print(f"Best epoch     : {best_epoch}")
            print(f"Best MainScore_raw : {best_main_score:.6f}")
            print(f"Best checkpoint: {best_ckpt_path}")
            print("=" * 70)
            break

    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help=(
            "Total target epoch number. When resuming, this is the FINAL epoch, "
            "not the number of additional epochs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["prompt", "random"],
        default="prompt",
        help=(
            "prompt: prompt-disjoint 8:2 split (current protocol); "
            "random: image-level random 8:2 split for benchmark comparison."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Example: outputs/AIGCIQA2023/last.pt",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience. Default: 8 epochs.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum MainScore improvement required to reset early stopping.",
    )

    args = parser.parse_args()

    df = read_metadata()
    dataset = AIGCIQA2023Dataset(train=True, dataframe=df)

    print(df.head(8).to_string(index=False))
    print("\\nFirst description:")
    first_name = str(df.iloc[0]["image_name"])
    print(dataset.description_map[first_name])

    if args.check_only:
        x = dataset[0]
        print(
            "\\nCHECK OK:",
            x["image_name"],
            "official=", x["original_image_name"],
            "model=", x["generator_model"],
            "semantic=", tuple(x["semantic_image"].shape),
            "visual=", tuple(x["visual_image"].shape),
            "mos=", x["mos"].item(),
        )
        return

    train_model(
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_mode=args.split_mode,
        seed=args.seed,
        resume_path=args.resume,
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()