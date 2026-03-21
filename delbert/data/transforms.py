"""
Transforms for converting sparse fingerprint format to molecular token sequences.
"""

from typing import List, Dict, Any
import torch
from collections import Counter


def sparse_fp_to_tokens(
    fp_indices: List[int],
    fp_values: List[int],
    fp_type: str,
    token_format: str = "binary",
    nbits: int = 2048
) -> List[str]:
    """
    Convert sparse fingerprint (indices + values) to token sequence.

    Args:
        fp_indices: List of fingerprint bit indices
        fp_values: List of counts for each index
        fp_type: Fingerprint type (e.g., 'ECFP4', 'ATOMPAIR')
        token_format: "binary" (default) or "count"
            - binary: tokens are {fp_type}_{index} (counts ignored)
            - count: tokens are {fp_type}_{index}_{count}
        nbits: Number of bits for binary mode (indices >= nbits are skipped)

    Returns:
        List of tokens in format determined by token_format
    """
    tokens = []
    for idx, count in zip(fp_indices, fp_values):
        if count > 0:  # Skip zero counts
            if token_format == "binary":
                if idx >= nbits:
                    continue  # Skip out-of-range indices in binary mode
                token = f"{fp_type}_{idx}"
            else:  # count mode
                token = f"{fp_type}_{idx}_{count}"
            tokens.append(token)
    return tokens


def molecule_to_tokens(
    row: Dict[str, Any],
    fingerprint_types: List[str],
    return_segment_ids: bool = False,
    token_format: str = "binary",
    nbits: int = 2048
):
    """
    Convert a molecule row with sparse fingerprints to token sequence.

    Args:
        row: Dictionary containing molecule data with {fp}_indices and {fp}_values
        fingerprint_types: List of fingerprint types to use (e.g., ['ECFP4', 'ATOMPAIR'])
        return_segment_ids: If True, also return segment IDs for each token
        token_format: "binary" (default) or "count"
        nbits: Number of bits for binary mode

    Returns:
        If return_segment_ids=False: Combined token sequence
        If return_segment_ids=True: Tuple of (tokens, segment_ids)
            segment_ids uses 1-indexed FP types (0 reserved for padding)
    """
    all_tokens = []
    all_segment_ids = [] if return_segment_ids else None

    for fp_idx, fp_type in enumerate(fingerprint_types):
        # Column names are {FP_TYPE}_indices and {FP_TYPE}_values
        # e.g., ECFP4_indices, ECFP4_values
        indices_key = f"{fp_type}_indices"
        values_key = f"{fp_type}_values"

        if indices_key in row and values_key in row:
            fp_indices = row[indices_key]
            fp_values = row[values_key]

            # Skip if empty
            if fp_indices is None or fp_values is None:
                continue

            # Convert to list if needed
            if not isinstance(fp_indices, list):
                fp_indices = fp_indices.tolist()
            if not isinstance(fp_values, list):
                fp_values = fp_values.tolist()

            fp_tokens = sparse_fp_to_tokens(fp_indices, fp_values, fp_type, token_format, nbits)
            all_tokens.extend(fp_tokens)

            # Segment ID: fp_idx + 1 (0 reserved for padding)
            if return_segment_ids:
                all_segment_ids.extend([fp_idx + 1] * len(fp_tokens))

    if return_segment_ids:
        return all_tokens, all_segment_ids
    return all_tokens


def build_vocabulary(dataset, fingerprint_types: List[str], min_frequency: int = 100, num_proc: int = None, sample_fraction: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """
    Build vocabulary from dataset with minimum frequency filtering.

    This function is used for count-based token format.
    For binary token format, use build_binary_vocabulary() instead.

    Args:
        dataset: HuggingFace dataset
        fingerprint_types: List of fingerprint types to use
        min_frequency: Minimum token frequency for inclusion
        num_proc: Number of processes for multiprocessing
        sample_fraction: Fraction of dataset to sample for vocabulary building (default: 0.2)
        seed: Random seed for sampling

    Returns:
        Dictionary with token_to_id, id_to_token, vocab_size
    """
    print("Building vocabulary (count mode)...")

    # Sample dataset if sample_fraction < 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(dataset) * sample_fraction)
        print(f"Sampling {sample_fraction*100:.0f}% of dataset ({sample_size:,} / {len(dataset):,} molecules)")
        print("  Shuffling dataset...", end=" ", flush=True)
        dataset = dataset.shuffle(seed=seed)
        print("done. Selecting samples...", end=" ", flush=True)
        dataset = dataset.select(range(sample_size))
        print("done.")

    def extract_tokens_batched(examples):
        """Extract tokens from a batch of molecules."""
        batch_tokens = []
        batch_size = len(examples[list(examples.keys())[0]])

        for i in range(batch_size):
            # Extract single example from batch
            example = {key: examples[key][i] for key in examples.keys()}
            # Use count mode for count-based vocabulary building
            tokens = molecule_to_tokens(example, fingerprint_types, token_format="count")
            batch_tokens.append(tokens)

        return {'tokens': batch_tokens}

    # Use batched multiprocessing for maximum speed
    token_dataset = dataset.map(
        extract_tokens_batched,
        batched=True,
        batch_size=1000,
        desc="Extracting tokens for vocabulary (batched)",
        num_proc=num_proc
    )

    # Compute token length statistics per molecule (before truncation)
    import numpy as np
    token_lengths = [len(tokens) for tokens in token_dataset['tokens']]
    token_lengths = np.array(token_lengths)

    print("\n" + "=" * 50)
    print("Token Length Statistics (per molecule, before truncation)")
    print("=" * 50)
    print(f"  Molecules analyzed: {len(token_lengths):,}")
    print(f"  Min tokens:    {np.min(token_lengths):,}")
    print(f"  Max tokens:    {np.max(token_lengths):,}")
    print(f"  Mean tokens:   {np.mean(token_lengths):,.1f}")
    print(f"  Std tokens:    {np.std(token_lengths):,.1f}")
    print(f"  Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    {p}%: {np.percentile(token_lengths, p):,.0f}")
    print("=" * 50 + "\n")

    # Count all tokens efficiently using vectorized operations
    print("Counting token frequencies...")
    
    # Count all tokens using pandas
    import pandas as pd
    print("Counting tokens...")
    
    # Flatten all tokens and convert to pandas Series
    from itertools import chain
    all_tokens = list(chain.from_iterable(token_dataset['tokens']))
    
    # Use pandas value_counts (optimized C implementation)
    token_series = pd.Series(all_tokens)
    token_counts_df = token_series.value_counts()
    token_counts = dict(token_counts_df)
    
    print(f"Counted {len(all_tokens)} tokens with {len(token_counts)} unique tokens")
    
    # Filter by frequency
    filtered_tokens = {token: count for token, count in token_counts.items() 
                      if count >= min_frequency}
    
    print(f"Total unique tokens: {len(token_counts)}")
    print(f"Tokens after filtering (>={min_frequency}): {len(filtered_tokens)}")
    
    # Create vocabulary mappings
    vocab_tokens = sorted(filtered_tokens.keys())  # Sort for consistency
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    # Add special tokens
    special_tokens = ['<unk>', '<pad>', '<mask>', '<cls>', '<sep>']
    for special_token in special_tokens:
        if special_token not in token_to_id:
            new_id = len(token_to_id)
            token_to_id[special_token] = new_id
            id_to_token[new_id] = special_token
    
    vocab_size = len(token_to_id)
    print(f"Final vocabulary size: {vocab_size}")
    
    return {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'vocab_size': vocab_size,
        'token_counts': filtered_tokens
    }


def build_binary_vocabulary(
    fingerprint_types: List[str],
    nbits: int = 2048
) -> Dict[str, Any]:
    """
    Build fixed vocabulary for binary fingerprint tokens.

    Vocabulary is deterministic: len(fingerprint_types) * nbits tokens + special tokens.
    No data scanning required.

    Args:
        fingerprint_types: List of fingerprint types (e.g., ['ECFP4', 'FCFP6', 'ATOMPAIR', 'TOPTOR'])
        nbits: Number of bits per fingerprint type

    Returns:
        Dictionary with token_to_id, id_to_token, vocab_size, token_counts (empty)
    """
    print(f"Building binary vocabulary: {len(fingerprint_types)} FP types x {nbits} bits")

    token_to_id = {}

    # Add fingerprint tokens in deterministic order (sorted FP types)
    for fp_type in sorted(fingerprint_types):
        for bit_idx in range(nbits):
            token = f"{fp_type}_{bit_idx}"
            token_to_id[token] = len(token_to_id)

    # Add special tokens
    special_tokens = ['<unk>', '<pad>', '<mask>', '<cls>', '<sep>']
    for special_token in special_tokens:
        token_to_id[special_token] = len(token_to_id)

    # Build reverse mapping
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    vocab_size = len(token_to_id)
    fp_tokens = len(fingerprint_types) * nbits
    print(f"Binary vocabulary size: {vocab_size}")
    print(f"  FP tokens: {fp_tokens}")
    print(f"  Special tokens: {len(special_tokens)}")

    return {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'vocab_size': vocab_size,
        'token_counts': {}  # Empty for binary mode (no frequency data)
    }


def tokenize_sequence(tokens: List[str], token_to_id: Dict[str, int], max_length: int = 512) -> List[int]:
    """
    Convert token sequence to ID sequence with truncation.

    Args:
        tokens: List of molecular tokens
        token_to_id: Token to ID mapping
        max_length: Maximum sequence length

    Returns:
        List of token IDs
    """
    unk_id = token_to_id['<unk>']
    encoded = []

    for token in tokens[:max_length]:  # Truncate if needed
        encoded.append(token_to_id.get(token, unk_id))

    return encoded


def encode_tokens(
    tokens: List[str],
    token_to_id: Dict[str, int],
    max_length: int = 2048,
    segment_ids: List[int] = None
) -> Dict[str, List[int]]:
    """
    Encode tokens to IDs with truncation.

    Args:
        tokens: List of molecular tokens
        token_to_id: Token to ID mapping
        max_length: Maximum sequence length
        segment_ids: Optional segment IDs for each token

    Returns:
        Dict with 'input_ids' and optionally 'segment_ids'
    """
    unk_id = token_to_id.get('<unk>', 0)

    encoded = []
    for token in tokens[:max_length]:
        encoded.append(token_to_id.get(token, unk_id))

    result = {'input_ids': encoded}

    if segment_ids is not None:
        result['segment_ids'] = segment_ids[:max_length]

    return result


def shuffle_fingerprint_spans(
    input_ids: List[int],
    segment_ids: List[int],
) -> tuple:
    """
    Shuffle entire fingerprint spans (all tokens from one FP type stay together).

    Example:
        Input:  [ECFP4 tokens..., FCFP6 tokens..., ATOMPAIR tokens...]
        Output: [ATOMPAIR tokens..., ECFP4 tokens..., FCFP6 tokens...] (random order)

    IMPORTANT: segment_ids are shuffled WITH their tokens, so segment embeddings
    will correctly follow the shuffled spans.

    Args:
        input_ids: List of token IDs
        segment_ids: List of segment IDs (1-indexed, 0 = padding)

    Returns:
        Tuple of (shuffled_input_ids, shuffled_segment_ids)
    """
    import random
    import numpy as np

    input_ids = np.array(input_ids)
    segment_ids = np.array(segment_ids)

    # Find unique non-padding segments
    unique_segments = np.unique(segment_ids)
    valid_segments = unique_segments[unique_segments > 0]

    if len(valid_segments) <= 1:
        return input_ids.tolist(), segment_ids.tolist()

    # Extract spans for each segment
    spans = []
    for seg_id in valid_segments:
        mask = segment_ids == seg_id
        spans.append((input_ids[mask].copy(), segment_ids[mask].copy()))

    # Shuffle span order
    random.shuffle(spans)

    # Reconstruct sequences
    shuffled_input_ids = np.concatenate([s[0] for s in spans])
    shuffled_segment_ids = np.concatenate([s[1] for s in spans])

    return shuffled_input_ids.tolist(), shuffled_segment_ids.tolist()


class MolecularCollator:
    """Data collator for molecular sequences with dynamic padding."""

    # Metadata fields to preserve (as lists, not tensors)
    METADATA_FIELDS = ['compound_id', 'library_id']

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int = None,
        mlm_probability: float = 0.15,
        vocab_size: int = None,
        segment_mlm_probability: float = 0.0,
        span_shuffle_probability: float = 0.0
    ):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mlm_probability = mlm_probability
        self.vocab_size = vocab_size
        self.segment_mlm_probability = segment_mlm_probability
        self.span_shuffle_probability = span_shuffle_probability

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with padding and optional masking.

        Returns a dict with:
            - input_ids: Padded token IDs (tensor)
            - attention_mask: Attention mask (tensor)
            - segment_ids: Segment IDs for each token (tensor, optional)
            - labels: Classification labels or MLM labels (tensor, optional)
            - compound_id: List of compound IDs (list, optional)
            - library_id: List of library IDs (list, optional)
        """
        # Extract sequences
        sequences = [item['input_ids'] for item in batch]
        labels = [item.get('labels') for item in batch]

        # Check if segment_ids present
        has_segment_ids = 'segment_ids' in batch[0]

        # Apply span shuffling (before padding)
        if self.span_shuffle_probability > 0.0 and has_segment_ids:
            import random
            segment_ids_list = [item['segment_ids'] for item in batch]
            shuffled_sequences = []
            shuffled_segment_ids_list = []

            for i, (seq, seg_ids) in enumerate(zip(sequences, segment_ids_list)):
                if random.random() < self.span_shuffle_probability:
                    shuffled_seq, shuffled_seg = shuffle_fingerprint_spans(seq, seg_ids)
                    shuffled_sequences.append(shuffled_seq)
                    shuffled_segment_ids_list.append(shuffled_seg)
                else:
                    shuffled_sequences.append(seq)
                    shuffled_segment_ids_list.append(seg_ids)

            sequences = shuffled_sequences
            # Update batch items for later segment_ids padding
            for i, item in enumerate(batch):
                item['input_ids'] = sequences[i]
                item['segment_ids'] = shuffled_segment_ids_list[i]

        # Extract metadata fields (preserve as lists, not tensors)
        metadata = {}
        for field in self.METADATA_FIELDS:
            values = [item.get(field) for item in batch]
            # Only include if at least one item has this field
            if any(v is not None for v in values):
                metadata[field] = values

        # Find max length in batch
        max_length = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        # Pad sequences
        padded_sequences = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            padded_sequences[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        batch_dict = {
            'input_ids': padded_sequences,
            'attention_mask': attention_mask
        }

        # Add segment_ids if present (for segment-based attention)
        if has_segment_ids:
            segment_ids_list = [item['segment_ids'] for item in batch]
            # Pad with 0 (padding segment)
            padded_segment_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
            for i, seg_ids in enumerate(segment_ids_list):
                seg_len = len(seg_ids)
                padded_segment_ids[i, :seg_len] = torch.tensor(seg_ids, dtype=torch.long)
            batch_dict['segment_ids'] = padded_segment_ids

        # Add labels if present (for classification)
        if labels[0] is not None:
            # Use long dtype for classification (CrossEntropyLoss expects class indices)
            batch_dict['labels'] = torch.tensor([label for label in labels], dtype=torch.long)

        # Add metadata fields (as lists, not tensors)
        batch_dict.update(metadata)

        # Apply MLM masking if mask token provided
        if self.mask_token_id is not None:
            batch_dict = self._apply_mlm_masking(batch_dict)

        return batch_dict
    
    def _apply_mlm_masking(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply MLM masking to input sequences.

        With probability segment_mlm_probability, masks all tokens from one randomly
        selected fingerprint segment. Otherwise, uses standard random token masking.
        Both modes apply the standard 80/10/10 split (80% [MASK], 10% keep, 10% random).
        """
        input_ids = batch_dict['input_ids'].clone()
        attention_mask = batch_dict['attention_mask']
        batch_size, seq_len = input_ids.shape

        # Create MLM labels (-100 for tokens we don't predict)
        mlm_labels = input_ids.clone()

        # Per-sample decision: segment masking vs token masking
        use_segment_masking = torch.zeros(batch_size, dtype=torch.bool)

        if self.segment_mlm_probability > 0:
            # Validation: segment_ids must be present
            if 'segment_ids' not in batch_dict:
                raise ValueError(
                    f"segment_mlm_probability={self.segment_mlm_probability} > 0, but 'segment_ids' "
                    f"not found in batch. Either set segment_mlm_probability=0 or ensure your "
                    f"data includes segment_ids (set return_segment_ids=true in data prep). "
                    f"Available keys: {list(batch_dict.keys())}"
                )
            use_segment_masking = torch.rand(batch_size) < self.segment_mlm_probability

        # Initialize masked_indices tensor
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # --- SEGMENT MASKING (for samples where use_segment_masking is True) ---
        if use_segment_masking.any():
            segment_ids = batch_dict['segment_ids']

            for i in range(batch_size):
                if use_segment_masking[i]:
                    # Get unique segments in this sample (excluding padding segment 0)
                    sample_segment_ids = segment_ids[i]
                    unique_segments = torch.unique(sample_segment_ids)
                    valid_segments = unique_segments[unique_segments > 0]

                    if len(valid_segments) > 0:
                        # Randomly select one segment to mask
                        chosen_segment = valid_segments[torch.randint(len(valid_segments), (1,)).item()]
                        # Mask all tokens in this segment
                        masked_indices[i] = (sample_segment_ids == chosen_segment)

        # --- TOKEN MASKING (for samples where use_segment_masking is False) ---
        if (~use_segment_masking).any():
            # Create probability matrix for token-level masking
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
            # Don't mask padding tokens
            probability_matrix.masked_fill_(attention_mask == 0, 0.0)
            # Zero out probability for segment-masked samples (already handled above)
            probability_matrix[use_segment_masking] = 0.0

            # Apply token-level masking via Bernoulli sampling
            token_masked_indices = torch.bernoulli(probability_matrix).bool()
            masked_indices = masked_indices | token_masked_indices

        # Set labels: -100 for unmasked tokens (don't compute loss on them)
        mlm_labels[~masked_indices] = -100

        # --- Apply 80/10/10 split to ALL masked tokens (same for both modes) ---
        # 80% of the time, replace with mask token
        mask_token_indices = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[mask_token_indices] = self.mask_token_id

        # 10% of the time, keep original (already in input_ids, do nothing)

        # 10% of the time, replace with random token
        random_token_indices = (torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() &
                               masked_indices & ~mask_token_indices)
        if self.vocab_size is not None:
            random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
            input_ids[random_token_indices] = random_words[random_token_indices]

        batch_dict['input_ids'] = input_ids
        batch_dict['labels'] = mlm_labels

        return batch_dict