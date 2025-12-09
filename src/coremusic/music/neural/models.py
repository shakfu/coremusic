#!/usr/bin/env python3
"""Pre-built neural network model architectures for music generation.

This module provides factory functions for creating various neural network
architectures suitable for music sequence learning:

- create_mlp_model: Feedforward MLP for fixed-window prediction
- create_rnn_model: Simple RNN for sequence modeling
- create_lstm_model: LSTM for long-term sequence modeling
- create_gru_model: GRU (simplified LSTM alternative)
- ModelFactory: High-level factory class

Example:
    >>> from coremusic.music.neural.models import create_lstm_model
    >>>
    >>> model = create_lstm_model(
    ...     input_size=128,
    ...     hidden_size=256,
    ...     output_size=128
    ... )
    >>> model.train(x_train, y_train)
"""

from typing import List, Optional

from coremusic.kann import (
    COST_MULTI_CROSS_ENTROPY,
    NeuralNetwork,
    RNN_NORM,
    RNN_VAR_H0,
)

from .data import BaseEncoder

# Re-export RNN flags for convenience
__all__ = [
    "create_mlp_model",
    "create_rnn_model",
    "create_lstm_model",
    "create_gru_model",
    "ModelFactory",
    "RNN_NORM",
    "RNN_VAR_H0",
]

# ============================================================================
# Model Factory Functions
# ============================================================================


def create_mlp_model(
    input_size: int,
    hidden_sizes: Optional[List[int]] = None,
    output_size: int = 128,
    dropout: float = 0.3,
    cost_type: int = COST_MULTI_CROSS_ENTROPY,
) -> NeuralNetwork:
    """Create a multi-layer perceptron for next-note prediction.

    MLPs work on a fixed input window (flattened one-hot sequence) and
    predict the next token. Simple but limited - no temporal awareness
    beyond the fixed window.

    Args:
        input_size: Size of input vector (seq_length * vocab_size for one-hot)
        hidden_sizes: List of hidden layer sizes (default: [512, 256])
        output_size: Size of output (vocabulary size)
        dropout: Dropout rate (0.0 = no dropout)
        cost_type: Loss function (default: multi-class cross-entropy)

    Returns:
        NeuralNetwork instance ready for training

    Example:
        >>> model = create_mlp_model(
        ...     input_size=32 * 128,  # 32 timesteps * 128 vocab
        ...     hidden_sizes=[512, 256],
        ...     output_size=128
        ... )
    """
    if hidden_sizes is None:
        hidden_sizes = [512, 256]

    return NeuralNetwork.mlp(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        cost_type=cost_type,
        dropout=dropout,
    )


def create_rnn_model(
    input_size: int,
    hidden_size: int = 256,
    output_size: int = 128,
    cost_type: int = COST_MULTI_CROSS_ENTROPY,
    rnn_flags: int = 0,
) -> NeuralNetwork:
    """Create a vanilla RNN for sequence modeling.

    Simple RNN with one recurrent layer. Good for learning short-term
    patterns but limited by vanishing gradients for longer sequences.

    Args:
        input_size: Size of input at each timestep (vocab_size for one-hot)
        hidden_size: Size of RNN hidden state
        output_size: Size of output (vocabulary size)
        cost_type: Loss function (default: multi-class cross-entropy)
        rnn_flags: RNN configuration flags (RNN_VAR_H0, RNN_NORM)

    Returns:
        NeuralNetwork instance

    Example:
        >>> model = create_rnn_model(
        ...     input_size=128,
        ...     hidden_size=256,
        ...     output_size=128
        ... )
    """
    return NeuralNetwork.rnn(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        cost_type=cost_type,
        rnn_flags=rnn_flags,
    )


def create_lstm_model(
    input_size: int,
    hidden_size: int = 512,
    output_size: int = 128,
    cost_type: int = COST_MULTI_CROSS_ENTROPY,
    rnn_flags: int = 0,
) -> NeuralNetwork:
    """Create an LSTM network for long-term sequence modeling.

    LSTM (Long Short-Term Memory) networks have gating mechanisms that
    allow them to learn long-range dependencies in sequences. Better
    gradient flow than vanilla RNNs.

    WARNING: LSTM training currently uses the feedforward training API
    (kann_train_fnn1) which does not properly implement backpropagation
    through time (BPTT). For effective sequence learning, use MLP models
    instead until proper RNN training is implemented. LSTM models created
    here can still be used for inference if trained with external tools.

    Args:
        input_size: Size of input at each timestep (vocab_size for one-hot)
        hidden_size: Size of LSTM hidden state
        output_size: Size of output (vocabulary size)
        cost_type: Loss function (default: multi-class cross-entropy)
        rnn_flags: RNN configuration flags (RNN_VAR_H0, RNN_NORM)

    Returns:
        NeuralNetwork instance

    Example:
        >>> model = create_lstm_model(
        ...     input_size=128,
        ...     hidden_size=512,
        ...     output_size=128
        ... )
    """
    return NeuralNetwork.lstm(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        cost_type=cost_type,
        rnn_flags=rnn_flags,
    )


def create_gru_model(
    input_size: int,
    hidden_size: int = 512,
    output_size: int = 128,
    cost_type: int = COST_MULTI_CROSS_ENTROPY,
    rnn_flags: int = 0,
) -> NeuralNetwork:
    """Create a GRU network for sequence modeling.

    GRU (Gated Recurrent Unit) is a simplified alternative to LSTM with
    fewer parameters. Often achieves similar performance with faster
    training.

    WARNING: GRU training currently uses the feedforward training API
    (kann_train_fnn1) which does not properly implement backpropagation
    through time (BPTT). For effective sequence learning, use MLP models
    instead until proper RNN training is implemented. GRU models created
    here can still be used for inference if trained with external tools.

    Args:
        input_size: Size of input at each timestep (vocab_size for one-hot)
        hidden_size: Size of GRU hidden state
        output_size: Size of output (vocabulary size)
        cost_type: Loss function (default: multi-class cross-entropy)
        rnn_flags: RNN configuration flags (RNN_VAR_H0, RNN_NORM)

    Returns:
        NeuralNetwork instance

    Example:
        >>> model = create_gru_model(
        ...     input_size=128,
        ...     hidden_size=512,
        ...     output_size=128
        ... )
    """
    return NeuralNetwork.gru(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        cost_type=cost_type,
        rnn_flags=rnn_flags,
    )


# ============================================================================
# Model Factory Class
# ============================================================================


class ModelFactory:
    """Factory for creating pre-configured neural network models.

    Provides a high-level interface for creating models appropriate
    for a given encoder and sequence length.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create(
        ...     model_type='lstm',
        ...     encoder=encoder,
        ...     seq_length=32,
        ...     hidden_size=256
        ... )
    """

    # Supported model types
    MODEL_TYPES = ("mlp", "rnn", "lstm", "gru")

    @staticmethod
    def create(
        model_type: str,
        encoder: BaseEncoder,
        seq_length: int,
        hidden_size: int = 256,
        dropout: float = 0.0,
        cost_type: int = COST_MULTI_CROSS_ENTROPY,
        rnn_flags: int = 0,
        **kwargs,
    ) -> NeuralNetwork:
        """Create a neural network model for the given encoder and configuration.

        Args:
            model_type: Type of model ('mlp', 'rnn', 'lstm', 'gru')
            encoder: The encoder used for tokenization (determines vocab_size)
            seq_length: Sequence length used for training
            hidden_size: Hidden layer size (or list for MLP)
            dropout: Dropout rate (MLP only)
            cost_type: Loss function type
            rnn_flags: RNN configuration flags (RNN models only)
            **kwargs: Additional arguments passed to model factory

        Returns:
            NeuralNetwork instance

        Raises:
            ValueError: If model_type is not supported
        """
        model_type = model_type.lower()
        vocab_size = encoder.vocab_size

        if model_type == "mlp":
            # MLP takes flattened one-hot input
            input_size = seq_length * vocab_size
            hidden_sizes = kwargs.get("hidden_sizes", [hidden_size, hidden_size // 2])
            return create_mlp_model(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=vocab_size,
                dropout=dropout,
                cost_type=cost_type,
            )

        elif model_type == "rnn":
            return create_rnn_model(
                input_size=vocab_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                cost_type=cost_type,
                rnn_flags=rnn_flags,
            )

        elif model_type == "lstm":
            return create_lstm_model(
                input_size=vocab_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                cost_type=cost_type,
                rnn_flags=rnn_flags,
            )

        elif model_type == "gru":
            return create_gru_model(
                input_size=vocab_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                cost_type=cost_type,
                rnn_flags=rnn_flags,
            )

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: {ModelFactory.MODEL_TYPES}"
            )

    @staticmethod
    def get_recommended_config(model_type: str, dataset_size: int) -> dict:
        """Get recommended hyperparameters based on model type and dataset size.

        Args:
            model_type: Type of model ('mlp', 'rnn', 'lstm', 'gru')
            dataset_size: Approximate number of training samples

        Returns:
            Dictionary of recommended hyperparameters
        """
        model_type = model_type.lower()

        # Base configuration
        config = {
            "learning_rate": 0.001,
            "mini_batch_size": 64,
            "max_epochs": 100,
            "validation_fraction": 0.1,
        }

        # Adjust based on dataset size
        if dataset_size < 1000:
            config["mini_batch_size"] = 16
            config["max_epochs"] = 200
            config["learning_rate"] = 0.0005
        elif dataset_size < 10000:
            config["mini_batch_size"] = 32
            config["max_epochs"] = 100
        else:
            config["mini_batch_size"] = 64
            config["max_epochs"] = 50

        # Model-specific adjustments
        if model_type == "mlp":
            config["hidden_size"] = 512
            config["dropout"] = 0.3
        elif model_type in ("lstm", "gru"):
            config["hidden_size"] = 256
            # Use layer normalization for better gradient flow
            config["rnn_flags"] = RNN_NORM
        elif model_type == "rnn":
            config["hidden_size"] = 128
            config["rnn_flags"] = RNN_NORM

        return config

    @staticmethod
    def list_models() -> List[str]:
        """List all supported model types.

        Returns:
            List of model type names
        """
        return list(ModelFactory.MODEL_TYPES)

    @staticmethod
    def describe_model(model_type: str) -> str:
        """Get a description of a model type.

        Args:
            model_type: Model type name

        Returns:
            Description string
        """
        descriptions = {
            "mlp": (
                "Multi-Layer Perceptron (MLP)\n"
                "- Fixed input window (flattened one-hot sequence)\n"
                "- Simple, fast training - RECOMMENDED\n"
                "- Limited temporal awareness (fixed window only)\n"
                "- Best for: Music generation, quick experiments"
            ),
            "rnn": (
                "Simple Recurrent Neural Network (RNN)\n"
                "- Processes sequences step-by-step\n"
                "- Basic recurrent connections\n"
                "- Prone to vanishing gradients\n"
                "- WARNING: Training not fully implemented (use MLP instead)"
            ),
            "lstm": (
                "Long Short-Term Memory (LSTM)\n"
                "- Gated recurrent architecture\n"
                "- Better gradient flow in theory\n"
                "- Can learn long-range dependencies\n"
                "- WARNING: Training not fully implemented (use MLP instead)\n"
                "- Note: Requires proper BPTT via kann_unroll()"
            ),
            "gru": (
                "Gated Recurrent Unit (GRU)\n"
                "- Simplified LSTM alternative\n"
                "- Fewer parameters, faster training\n"
                "- Similar performance to LSTM\n"
                "- WARNING: Training not fully implemented (use MLP instead)"
            ),
        }
        model_type = model_type.lower()
        if model_type not in descriptions:
            raise ValueError(f"Unknown model type: {model_type}")
        return descriptions[model_type]
