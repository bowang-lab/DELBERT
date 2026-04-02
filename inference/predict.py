#!/usr/bin/env python3
"""
Run inference with a DELBERT model from HuggingFace Hub or local directory.

Accepts molecular fingerprints as dense arrays (from AIRCHECK parquet files)
or as Python dicts, and returns activity predictions.

Usage:
    # Predict from a parquet file (batch)
    python deploy/scripts/predict.py \
        --model wanglab/delbert-wdr91 \
        --parquet data/WDR91/WDR91.parquet \
        --output predictions.csv \
        --batch_size 100

    # Predict from a local model directory
    python deploy/scripts/predict.py \
        --model /path/to/local/model \
        --local \
        --parquet data/WDR91/WDR91.parquet

Python API:
    from inference.predict import predict, predict_from_parquet

    # Single molecule (dict with dense FP arrays of length 2048)
    probs = predict(
        {"ECFP4": [...], "FCFP6": [...], "ATOMPAIR": [...], "TOPTOR": [...]},
        model_path="wanglab/delbert-wdr91",
    )

    # Batch from parquet
    df = predict_from_parquet(
        "data/WDR91/WDR91.parquet",
        model_path="wanglab/delbert-wdr91",
    )
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Add project root to path

from delbert.data.tokenizer import MolecularTokenizer
from delbert.data.transforms import MolecularCollator, molecule_to_tokens
from delbert.models.delbert_model import DELBERTConfig, DELBERTForSequenceClassification

FINGERPRINT_TYPES = ["ECFP4", "FCFP6", "ATOMPAIR", "TOPTOR"]

# The deployed model uses count-format tokens (e.g., "ECFP4_41_1", "FCFP6_0_12")
# where the third component is the fingerprint count. This must always be "count"
# regardless of what tokenizer_config.json reports.
TOKEN_FORMAT = "count"


# ---------------------------------------------------------------------------
# Dense → sparse conversion
# ---------------------------------------------------------------------------

def dense_to_sparse(dense_array) -> Tuple[List[int], List[int]]:
    """
    Convert a dense fingerprint array to sparse (indices, values).

    Args:
        dense_array: Array-like of length 2048 (int counts, 0 = unset)

    Returns:
        (indices, values) — lists of nonzero positions and their counts
    """
    arr = np.asarray(dense_array, dtype=np.int32)
    nonzero = np.nonzero(arr)[0]
    return nonzero.tolist(), arr[nonzero].tolist()


def dense_dict_to_sparse_row(
    molecule: Dict[str, list],
    fingerprint_types: List[str] = FINGERPRINT_TYPES,
) -> Dict[str, list]:
    """
    Convert a dict with dense FP arrays to the sparse row format
    expected by molecule_to_tokens().

    Args:
        molecule: Dict mapping FP type names to dense arrays (length 2048).
                  Example: {"ECFP4": [0,0,1,...], "FCFP6": [...], ...}
        fingerprint_types: Which FP types to include

    Returns:
        Dict with "{FP}_indices" and "{FP}_values" keys for each FP type.
    """
    row = {}
    for fp_type in fingerprint_types:
        if fp_type not in molecule:
            raise ValueError(
                f"Missing fingerprint '{fp_type}' in molecule dict. "
                f"Required: {fingerprint_types}. Got: {list(molecule.keys())}"
            )
        indices, values = dense_to_sparse(molecule[fp_type])
        row[f"{fp_type}_indices"] = indices
        row[f"{fp_type}_values"] = values
    return row


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    from_hub: bool = True,
    device: Optional[str] = None,
) -> Tuple[DELBERTForSequenceClassification, MolecularTokenizer, torch.device]:
    """
    Load a DELBERT model and tokenizer.

    Args:
        model_path: HuggingFace repo ID (e.g., "wanglab/delbert-wdr91")
                    or local directory path
        from_hub: If True, download from HF Hub. If False, load from local dir.
        device: Device string ("cuda", "cpu", etc.). Auto-detected if None.

    Returns:
        (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Resolve model directory
    if from_hub:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_path)
    else:
        local_dir = model_path

    local_dir = Path(local_dir)

    # Load config
    with open(local_dir / "config.json") as f:
        config_dict = json.load(f)

    num_labels = config_dict.pop("num_labels", 2)
    # Remove non-config fields
    for key in ["architectures", "model_type", "lora_merged"]:
        config_dict.pop(key, None)

    config = DELBERTConfig(**config_dict)
    model = DELBERTForSequenceClassification(config, num_labels=num_labels)

    # Load weights (safetensors preferred, pytorch_model.bin fallback)
    safetensors_path = local_dir / "model.safetensors"
    pytorch_path = local_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(str(pytorch_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights in {local_dir}. "
            f"Expected 'model.safetensors' or 'pytorch_model.bin'."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = MolecularTokenizer.from_pretrained(str(local_dir))

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def prepare_molecule(
    molecule: Dict[str, list],
    tokenizer: MolecularTokenizer,
    fingerprint_types: List[str] = FINGERPRINT_TYPES,
) -> Dict[str, list]:
    """
    Convert a molecule dict (dense FP arrays) to model-ready token IDs.

    Args:
        molecule: Dict with dense FP arrays. Keys must include all
                  fingerprint_types (e.g., "ECFP4", "FCFP6", ...).
                  Each value is an array-like of length 2048.
        tokenizer: Loaded MolecularTokenizer
        fingerprint_types: FP types to tokenize

    Returns:
        Dict with "input_ids" (List[int]) and "segment_ids" (List[int])
    """
    sparse_row = dense_dict_to_sparse_row(molecule, fingerprint_types)
    tokens, segment_ids = molecule_to_tokens(
        sparse_row,
        fingerprint_types,
        return_segment_ids=True,
        token_format=TOKEN_FORMAT,
    )
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return {"input_ids": input_ids, "segment_ids": segment_ids}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    prepared_samples: List[Dict[str, list]],
    model: DELBERTForSequenceClassification,
    tokenizer: MolecularTokenizer,
    device: torch.device,
    batch_size: int = 100,
) -> np.ndarray:
    """
    Run model inference on prepared samples.

    Args:
        prepared_samples: List of dicts with "input_ids" and "segment_ids"
        model: Loaded model (on device, in eval mode)
        tokenizer: Tokenizer (for pad_token_id)
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Array of positive-class probabilities, shape (n_samples,)
    """
    collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    all_probs = []

    for start in range(0, len(prepared_samples), batch_size):
        batch_items = prepared_samples[start : start + batch_size]
        batch = collator(batch_items)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            segment_ids = segment_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )
        probs = (
            torch.softmax(outputs["logits"], dim=-1)[:, 1]
            .float()
            .cpu()
            .numpy()
        )
        all_probs.append(probs)

    return np.concatenate(all_probs)


def predict(
    molecules: Union[Dict[str, list], List[Dict[str, list]]],
    model_path: str = "wanglab/delbert-wdr91",
    from_hub: bool = True,
    device: Optional[str] = None,
    batch_size: int = 100,
) -> List[float]:
    """
    Predict activity from fingerprint vectors.

    Args:
        molecules: A single molecule dict or list of dicts.
                   Each dict maps FP type names to dense arrays of length 2048:
                   {"ECFP4": [...], "FCFP6": [...], "ATOMPAIR": [...], "TOPTOR": [...]}
        model_path: HF repo ID or local directory
        from_hub: If True, download from HF Hub
        device: Device string (auto-detected if None)
        batch_size: Batch size for inference

    Returns:
        List of positive-class probabilities (one per molecule)
    """
    if isinstance(molecules, dict):
        molecules = [molecules]

    model, tokenizer, dev = load_model(model_path, from_hub=from_hub, device=device)

    prepared = [prepare_molecule(m, tokenizer) for m in molecules]
    probs = _run_inference(prepared, model, tokenizer, dev, batch_size)
    return probs.tolist()


def predict_from_parquet(
    parquet_path: str,
    model_path: str = "wanglab/delbert-wdr91",
    from_hub: bool = True,
    device: Optional[str] = None,
    batch_size: int = 100,
):
    """
    Predict activity for all molecules in a parquet file.

    The parquet must contain dense FP array columns: ECFP4, FCFP6, ATOMPAIR, TOPTOR.
    Optionally includes COMPOUND_ID for identification.

    Args:
        parquet_path: Path to AIRCHECK parquet file
        model_path: HF repo ID or local directory
        from_hub: If True, download from HF Hub
        device: Device string (auto-detected if None)
        batch_size: Batch size for inference

    Returns:
        pandas DataFrame with columns: compound_id (if available), probability
    """
    import pandas as pd

    # Load parquet
    columns_to_load = list(FINGERPRINT_TYPES)
    df = pd.read_parquet(parquet_path)

    # Validate required columns
    missing = [c for c in FINGERPRINT_TYPES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet missing required FP columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    has_compound_id = "COMPOUND_ID" in df.columns

    # Load model once
    model, tokenizer, dev = load_model(model_path, from_hub=from_hub, device=device)

    # Prepare all molecules
    print(f"Preparing {len(df)} molecules...", flush=True)
    prepared = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        molecule = {fp: row[fp] for fp in FINGERPRINT_TYPES}
        prepared.append(prepare_molecule(molecule, tokenizer))

    # Run inference
    print(f"Running inference (batch_size={batch_size})...", flush=True)
    probs = _run_inference(prepared, model, tokenizer, dev, batch_size)

    # Build result DataFrame
    result = pd.DataFrame({"probability": probs})
    if has_compound_id:
        result.insert(0, "compound_id", df["COMPOUND_ID"].values)

    print(f"Done. {len(result)} predictions.", flush=True)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run DELBERT inference on molecular fingerprints"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo ID (e.g., wanglab/delbert-wdr91) or local path",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Load model from local directory instead of HF Hub",
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to AIRCHECK parquet file with FP columns",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Inference batch size (default: 100)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: auto-detect cuda/cpu)",
    )

    args = parser.parse_args()

    result = predict_from_parquet(
        parquet_path=args.parquet,
        model_path=args.model,
        from_hub=not args.local,
        device=args.device,
        batch_size=args.batch_size,
    )

    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
