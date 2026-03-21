"""
Molecular tokenizer using standard HuggingFace API.
"""

import json
import os
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer


class MolecularTokenizer(PreTrainedTokenizer):
    """
    Molecular fingerprint tokenizer that uses standard HuggingFace API.
    
    This is not a text tokenizer - it maps pre-computed molecular fingerprint
    features to IDs for use in transformer models.
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
    }
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        cls_token: str = "<cls>",
        sep_token: str = "<sep>",
        fingerprint_types: List[str] = None,
        token_format: str = "binary",
        fingerprint_nbits: int = 2048,
        **kwargs
    ):
        # Store molecular tokenizer attributes
        self.fingerprint_types = fingerprint_types
        self.token_format = token_format
        self.fingerprint_nbits = fingerprint_nbits

        # Load vocabulary if file provided
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._token_to_id = json.load(f)
        else:
            # Initialize empty if no vocab file
            self._token_to_id = {}

        # Build reverse mapping
        self._id_to_token = {int(v): k for k, v in self._token_to_id.items()}

        # Initialize parent class
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            fingerprint_types=fingerprint_types,
            token_format=token_format,
            fingerprint_nbits=fingerprint_nbits,
            **kwargs
        )
    
    @classmethod
    def from_vocabulary(cls, token_to_id: Dict[str, int], **kwargs):
        """
        Create tokenizer from a vocabulary dictionary.
        
        Args:
            token_to_id: Dictionary mapping tokens to IDs
            **kwargs: Additional tokenizer arguments
            
        Returns:
            MolecularTokenizer instance
        """
        # Create temporary vocab file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(token_to_id, f)
            temp_vocab_file = f.name
        
        try:
            # Initialize with vocab file
            tokenizer = cls(vocab_file=temp_vocab_file, **kwargs)
            # Store the vocabulary internally
            tokenizer._token_to_id = token_to_id
            tokenizer._id_to_token = {int(v): k for k, v in token_to_id.items()}
            return tokenizer
        finally:
            # Clean up temp file
            if os.path.exists(temp_vocab_file):
                os.remove(temp_vocab_file)
    
    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        # Required by PreTrainedTokenizer
        vocab = self._token_to_id.copy()
        # Add special tokens if not in vocab
        for token in [self.unk_token, self.pad_token, self.mask_token, self.cls_token, self.sep_token]:
            if token and token not in vocab:
                vocab[token] = len(vocab)
        return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """
        'Tokenize' molecular features.
        Note: In our case, text is already space-separated tokens.
        
        Args:
            text: Space-separated molecular tokens
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID using vocabulary."""
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token using vocabulary."""
        return self._id_to_token.get(index, self.unk_token)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to ID(s)."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to token(s)."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id_) for id_ in ids]
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs by adding special tokens.
        For molecular models, we typically don't add special tokens as the 
        features are pre-computed.
        """
        # For molecular data, often we don't add CLS/SEP tokens
        # Modify this based on your model's needs
        return token_ids_0
    
    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get mask identifying special tokens.
        For molecular features, typically all zeros (no special tokens in sequences).
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, 
                token_ids_1=token_ids_1, 
                already_has_special_tokens=True
            )
        return [0] * len(token_ids_0)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save vocabulary files to directory (HuggingFace standard method).
        
        Args:
            save_directory: Directory to save vocabulary
            filename_prefix: Optional prefix for filename
            
        Returns:
            Tuple of saved files
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._token_to_id, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save tokenizer using standard HuggingFace API.
        This will create:
        - vocab.json (vocabulary)
        - special_tokens_map.json (special tokens) 
        - tokenizer_config.json (configuration)
        - added_tokens.json (if applicable)
        """
        # This calls save_vocabulary internally and handles all the other files
        return super().save_pretrained(save_directory, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Load tokenizer using standard HuggingFace API.
        
        Example:
            tokenizer = MolecularTokenizer.from_pretrained("path/to/tokenizer/")
        """
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# Additional utility functions for molecular-specific operations
def create_molecular_tokenizer(
    vocabulary: Dict[str, int],
    fingerprint_types: List[str] = None,
    min_frequency: int = None,
    token_format: str = "binary",
    fingerprint_nbits: int = 2048
) -> MolecularTokenizer:
    """
    Helper function to create a molecular tokenizer with metadata.

    Args:
        vocabulary: Token to ID mapping
        fingerprint_types: Types of fingerprints used
        min_frequency: Minimum token frequency used (count mode only)
        token_format: "binary" or "count"
        fingerprint_nbits: Number of bits per fingerprint type (binary mode)

    Returns:
        Configured MolecularTokenizer
    """
    tokenizer = MolecularTokenizer.from_vocabulary(
        vocabulary,
        fingerprint_types=fingerprint_types,
        token_format=token_format,
        fingerprint_nbits=fingerprint_nbits
    )

    # Add optional attributes (not in __init__, so set manually)
    if min_frequency:
        tokenizer.min_frequency = min_frequency
    tokenizer.model_type = "molecular_transformer"

    return tokenizer
