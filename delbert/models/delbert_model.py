"""
DELBERT: ModernBERT-based model for molecular fingerprint sequences.

Flat architecture with 2 task models:
- DELBERTForMLM: encoder + MLM head (pretraining)
- DELBERTForSequenceClassification: encoder + classifier (finetuning)

Both use ModernBertModel directly as .encoder, with optional segment embeddings
and segment attention handled via a shared _forward_encoder() helper.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput


def create_segment_attention_mask(
    segment_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create 4D attention mask for segment-restricted (Intra-FP) attention.

    Tokens can only attend to other tokens with the same segment_id.
    Padding tokens (segment_id=0) cannot attend or be attended to.

    Args:
        segment_ids: [batch, seq_len] - segment ID for each token (0 = padding)
        attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
        dtype: torch dtype for output (determines min value for masking)

    Returns:
        [batch, 1, seq_len, seq_len] attention mask
        Values: 0.0 where attention allowed, -inf where blocked
    """
    device = segment_ids.device

    # Same segment check: [batch, seq_len, seq_len]
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)

    # Exclude padding: padding tokens (segment_id=0) should not match anything
    not_padding_query = (segment_ids != 0).unsqueeze(2)  # [batch, seq_len, 1]
    not_padding_key = (segment_ids != 0).unsqueeze(1)    # [batch, 1, seq_len]

    # Can attend iff: same segment AND query not padding AND key not padding
    can_attend = same_segment & not_padding_query & not_padding_key

    # Also apply the original attention_mask (handles actual padding positions)
    valid_key = attention_mask.unsqueeze(1).bool()  # [batch, 1, seq_len]
    can_attend = can_attend & valid_key

    # Convert to float mask: 0 = attend, min_value = block
    min_value = torch.finfo(dtype).min
    attn_mask = torch.where(
        can_attend,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_value, dtype=dtype, device=device)
    )

    # Add head dimension: [batch, 1, seq_len, seq_len]
    return attn_mask.unsqueeze(1)


def compute_position_ids(
    segment_ids: torch.Tensor,
    seq_len: int,
    mode: str,
    device: torch.device
) -> torch.Tensor:
    """
    Compute position_ids based on segment structure and encoding mode.

    Args:
        segment_ids: [batch, seq_len] segment IDs (0 = padding)
        seq_len: sequence length
        mode: 'absolute', 'reset', or 'none'
        device: torch device

    Returns:
        position_ids: [batch, seq_len]
    """
    batch_size = segment_ids.shape[0]

    if mode == 'absolute':
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    elif mode == 'none':
        return torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    elif mode == 'reset':
        # Reset at each segment boundary
        # Example: segment_ids = [1,1,1,2,2,3,3,3]
        #          position_ids = [0,1,2,0,1,0,1,2]
        segment_changed = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            segment_ids[:, 1:] != segment_ids[:, :-1]
        ], dim=1)

        ones = torch.ones_like(segment_ids, dtype=torch.long)
        cumsum = torch.cumsum(ones, dim=1)
        boundary_cumsum = cumsum * segment_changed.long()
        reset_values, _ = torch.cummax(boundary_cumsum, dim=1)
        position_ids = cumsum - reset_values

        return position_ids

    else:
        raise ValueError(f"Unknown segment_position_encoding: {mode}")


class DELBERTConfig(ModernBertConfig):
    """
    Configuration class for DELBERT model.
    Extends ModernBertConfig with molecular-specific parameters.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 1152,
        max_position_embeddings: int = 1024,
        global_rope_theta: float = 160000.0,
        local_attention: int = 128,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        classifier_pooling: str = 'cls',
        pad_token_id: int = 0,
        use_segment_attention: bool = False,
        segment_position_encoding: str = 'absolute',
        use_segment_embeddings: bool = False,
        num_segment_types: int = 8,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            global_rope_theta=global_rope_theta,
            local_attention=local_attention,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout=classifier_dropout,
            classifier_pooling=classifier_pooling,
            pad_token_id=pad_token_id,
            **kwargs
        )
        self.use_segment_attention = use_segment_attention
        self.segment_position_encoding = segment_position_encoding
        self.use_segment_embeddings = use_segment_embeddings
        self.num_segment_types = num_segment_types


class SegmentEmbeddingLayer(nn.Module):
    """Learnable embeddings for fingerprint segment types."""

    def __init__(self, num_segment_types: int, hidden_size: int):
        super().__init__()
        self.segment_embeddings = nn.Embedding(
            num_embeddings=num_segment_types,
            embedding_dim=hidden_size,
            padding_idx=0
        )
        nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.segment_embeddings.weight[0].zero_()

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.segment_embeddings(segment_ids)


def apply_layer_types(
    encoder: nn.Module,
    layer_types: list,
    local_attention_window: int
) -> None:
    """
    Apply custom attention patterns to encoder layers.

    Args:
        encoder: ModernBertModel (or PEFT-wrapped). Must have .layers attribute.
        layer_types: List of 'full_attention' or 'sliding_attention' per layer
        local_attention_window: Window size for sliding attention
    """
    if len(layer_types) != len(encoder.layers):
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but model has "
            f"{len(encoder.layers)} layers"
        )

    half_window = local_attention_window // 2
    for i, (layer, layer_type) in enumerate(zip(encoder.layers, layer_types)):
        if layer_type == 'full_attention':
            layer.attn.local_attention = (-1, -1)
        elif layer_type == 'sliding_attention':
            layer.attn.local_attention = (half_window, half_window)
        else:
            raise ValueError(
                f"Unknown layer_type '{layer_type}' at index {i}. "
                f"Use 'full_attention' or 'sliding_attention'"
            )


def _forward_encoder(
    encoder: nn.Module,
    config: DELBERTConfig,
    segment_embedding_layer: Optional[SegmentEmbeddingLayer],
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
    output_hidden_states: Optional[bool] = None,
) -> BaseModelOutput:
    """
    Shared encoder forward handling segment embeddings and segment attention.

    Three paths:
    A) Standard: no segment features → plain encoder forward
    B) Segment embeddings only: enrich token embeddings, then standard forward
    C) Segment attention (± segment embeddings): manual layer loop with segment mask
    """
    use_segment_attention = getattr(config, 'use_segment_attention', False)

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_len), device=device, dtype=torch.long
        )

    # Compute position_ids based on segment structure
    position_ids = None
    if segment_ids is not None:
        mode = getattr(config, 'segment_position_encoding', 'absolute')
        position_ids = compute_position_ids(segment_ids, seq_len, mode, device)

    needs_segment_embeddings = (
        segment_embedding_layer is not None and segment_ids is not None
    )

    if use_segment_attention:
        # PATH C: Segment attention — manual layer loop
        if segment_ids is None:
            raise ValueError(
                "use_segment_attention=True but segment_ids is None. "
                "Ensure data was prepared with 'return_segment_ids: true'."
            )

        # Get standard global mask for full-attention layers
        global_attn_mask, _ = encoder._update_attention_mask(
            attention_mask, output_attentions=False
        )

        # Create segment-restricted mask for sliding-window layers
        encoder_dtype = encoder.embeddings.tok_embeddings.weight.dtype
        segment_mask = create_segment_attention_mask(
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            dtype=encoder_dtype,
        )

        # Get token embeddings
        hidden_states = encoder.embeddings(input_ids=input_ids)

        # Add segment embeddings if enabled
        if needs_segment_embeddings:
            hidden_states = hidden_states + segment_embedding_layer(segment_ids)

        all_hidden_states = () if output_hidden_states else None

        # Manual layer loop: global mask for full-attn layers,
        # segment mask for sliding-window layers
        for layer in encoder.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=global_attn_mask,
                sliding_window_mask=segment_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = encoder.final_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    elif needs_segment_embeddings:
        # PATH B: Segment embeddings only — enrich, then standard forward
        hidden_states = encoder.embeddings(input_ids=input_ids)
        hidden_states = hidden_states + segment_embedding_layer(segment_ids)

        return encoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    else:
        # PATH A: Standard — plain ModernBertModel forward
        return encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


class DELBERTForMLM(nn.Module):
    """
    DELBERT model for Masked Language Modeling (pretraining).

    Flat structure:
        .encoder = ModernBertModel
        .segment_embedding_layer = SegmentEmbeddingLayer (optional)
        .head = ModernBertPredictionHead
        .decoder = Linear
    """

    def __init__(self, config: DELBERTConfig):
        super().__init__()
        self.config = config
        self._use_segment_attention = getattr(config, 'use_segment_attention', False)

        # Encoder
        self.encoder = ModernBertModel(config)

        # Apply layer_types if present in config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types:
            apply_layer_types(self.encoder, layer_types, config.local_attention)

        # Optional segment embeddings
        if getattr(config, 'use_segment_embeddings', False):
            self.segment_embedding_layer = SegmentEmbeddingLayer(
                num_segment_types=getattr(config, 'num_segment_types', 8),
                hidden_size=config.hidden_size,
            )
        else:
            self.segment_embedding_layer = None

        # MLM head
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertPredictionHead,
        )
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self._use_segment_attention and segment_ids is None:
            raise ValueError(
                "DELBERTForMLM has use_segment_attention=True but segment_ids "
                "is None. Ensure data prep config has 'return_segment_ids: true'."
            )
        if self.segment_embedding_layer is not None and segment_ids is None:
            raise ValueError(
                "use_segment_embeddings=True but segment_ids is None. "
                "Ensure data prep has 'return_segment_ids: true'."
            )

        outputs = _forward_encoder(
            encoder=self.encoder,
            config=self.config,
            segment_embedding_layer=self.segment_embedding_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        hidden_states = outputs.last_hidden_state
        predictions = self.decoder(self.head(hidden_states))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                predictions.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            'loss': loss,
            'logits': predictions,
            'hidden_states': outputs.hidden_states,
        }

    def get_encoder(self):
        """Get the underlying ModernBERT encoder."""
        return self.encoder

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, config: Optional[DELBERTConfig] = None
    ):
        """
        Load model from Lightning .ckpt checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file
            config: Optional config (will load from checkpoint if not provided)

        Returns:
            DELBERTForMLM model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if config is None:
            if 'model_config' not in checkpoint:
                raise ValueError(
                    "No model_config found in checkpoint and no config provided"
                )
            config = DELBERTConfig(**checkpoint['model_config'])

        model = cls(config)

        # Lightning stores state under 'state_dict' with 'model.' prefix
        # (PretrainModel.model = DELBERTForMLM)
        state_dict = {
            k.replace('model.', '', 1): v
            for k, v in checkpoint['state_dict'].items()
        }
        model.load_state_dict(state_dict)

        # layer_types are already applied in __init__ (read from config)

        return model


class DELBERTForSequenceClassification(nn.Module):
    """
    DELBERT model for sequence classification (finetuning).

    Flat structure:
        .encoder = ModernBertModel
        .segment_embedding_layer = SegmentEmbeddingLayer (optional)
        .classifier = Linear
        .dropout = Dropout
    """

    def __init__(
        self,
        config: DELBERTConfig,
        num_labels: int = 2,
        pretrained_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self._use_segment_attention = getattr(config, 'use_segment_attention', False)

        # Encoder
        self.encoder = ModernBertModel(config)

        # Apply layer_types if present in config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types:
            apply_layer_types(self.encoder, layer_types, config.local_attention)

        # Optional segment embeddings
        if getattr(config, 'use_segment_embeddings', False):
            self.segment_embedding_layer = SegmentEmbeddingLayer(
                num_segment_types=getattr(config, 'num_segment_types', 8),
                hidden_size=config.hidden_size,
            )
        else:
            self.segment_embedding_layer = None

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout)

        if pretrained_encoder is not None:
            self.load_pretrained_encoder(pretrained_encoder)

    def load_pretrained_encoder(self, pretrained_model: 'DELBERTForMLM'):
        """Load encoder and segment embedding weights from pretrained MLM model."""
        self.encoder.load_state_dict(pretrained_model.encoder.state_dict())

        # Transfer segment embeddings if both models have them
        if self.segment_embedding_layer is not None and pretrained_model.segment_embedding_layer is not None:
            self.segment_embedding_layer.load_state_dict(
                pretrained_model.segment_embedding_layer.state_dict()
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self._use_segment_attention and segment_ids is None:
            raise ValueError(
                "DELBERTForSequenceClassification has use_segment_attention=True "
                "but segment_ids is None. Ensure data prep config has "
                "'return_segment_ids: true'."
            )
        if self.segment_embedding_layer is not None and segment_ids is None:
            raise ValueError(
                "use_segment_embeddings=True but segment_ids is None. "
                "Ensure data prep has 'return_segment_ids: true'."
            )

        outputs = _forward_encoder(
            encoder=self.encoder,
            config=self.config,
            segment_embedding_layer=self.segment_embedding_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        hidden_states = outputs.last_hidden_state

        # Pooling
        pooling = getattr(self.config, 'classifier_pooling', 'cls')
        if pooling == 'cls':
            pooled = hidden_states[:, 0]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
        }

    def get_encoder(self):
        """Get the underlying ModernBERT encoder."""
        return self.encoder

    @classmethod
    def from_pretrained_mlm(
        cls,
        mlm_checkpoint: str,
        num_labels: int = 2,
        config: Optional[DELBERTConfig] = None,
    ):
        """
        Initialize classification model from pretrained MLM checkpoint.

        Args:
            mlm_checkpoint: Path to MLM checkpoint
            num_labels: Number of classification labels
            config: Optional config (will load from checkpoint if not provided)

        Returns:
            DELBERTForSequenceClassification model
        """
        mlm_model = DELBERTForMLM.from_pretrained(mlm_checkpoint, config)

        if config is None:
            config = mlm_model.config

        model = cls(config, num_labels=num_labels)
        model.load_pretrained_encoder(mlm_model)

        return model
