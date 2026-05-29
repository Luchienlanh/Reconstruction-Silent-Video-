"""Model registry — factory function to build any supported model by name."""

from __future__ import annotations

from torch import nn


def build_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool = True,
    transformer_layers: int = 1,
    feature_set: str = "full",
) -> nn.Module:
    """Create a model instance by name.

    Args:
        model_type: One of 'lstm', 'speech_tcn', 'snn', 'pose_r2plus1d',
                     'stgcn', 'spiking_stgcn'.
        input_dim: Input feature dimension (132 for MediaPipe pose).
        num_classes: Number of output classes.
        hidden_size: Hidden dimension for the model.
        num_layers: Number of layers/blocks.
        dropout: Dropout probability.
        bidirectional: Whether LSTM uses bidirectional (LSTM only).
        transformer_layers: Number of transformer layers (SpeechTCN only).
        feature_set: Per-joint feature ablation for ST-GCN variants.

    Returns:
        An nn.Module ready for training.
    """
    model_type = model_type.lower()

    if model_type == "lstm":
        from .lstm import PoseLSTM

        return PoseLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    if model_type in {"speech_tcn", "tcn"}:
        from .speech_tcn import SpeechTCNPoseClassifier

        return SpeechTCNPoseClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            transformer_layers=transformer_layers,
        )

    if model_type == "snn":
        from .snn import SNNPoseClassifier

        return SNNPoseClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_type in {"pose_r2plus1d", "r2plus1d"}:
        from .pose_r2plus1d import PoseR2Plus1DClassifier

        return PoseR2Plus1DClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_type == "stgcn":
        from .spiking_stgcn import STGCN

        return STGCN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            feature_set=feature_set,
        )

    if model_type == "spiking_stgcn":
        from .spiking_stgcn import SpikingSTGCN

        return SpikingSTGCN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            feature_set=feature_set,
        )

    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        f"Choose from: lstm, speech_tcn, snn, pose_r2plus1d, stgcn, spiking_stgcn"
    )


MODEL_TYPES = ["lstm", "speech_tcn", "snn", "pose_r2plus1d", "stgcn", "spiking_stgcn"]
