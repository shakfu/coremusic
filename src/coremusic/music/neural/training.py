#!/usr/bin/env python3
"""Training utilities and callbacks for neural network music models.

This module provides:
- TrainingConfig: Configuration dataclass for training parameters
- Trainer: High-level training interface with progress reporting
- Callbacks: EarlyStopping, ModelCheckpoint, ProgressLogger, LearningRateScheduler

Example:
    >>> from coremusic.music.neural import Trainer, TrainingConfig
    >>> from coremusic.music.neural.training import EarlyStopping, ModelCheckpoint
    >>>
    >>> config = TrainingConfig(
    ...     learning_rate=0.001,
    ...     batch_size=32,
    ...     max_epochs=100
    ... )
    >>> trainer = Trainer(model, config)
    >>> trainer.add_callback(EarlyStopping(patience=10))
    >>> trainer.add_callback(ModelCheckpoint('best_model.kan'))
    >>> history = trainer.train(x_train, y_train)
"""

import logging
import time
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from coremusic.kann import NeuralNetwork

logger = logging.getLogger(__name__)


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for neural network training.

    Attributes:
        learning_rate: Learning rate for optimizer (RMSprop)
        batch_size: Mini-batch size
        max_epochs: Maximum number of training epochs
        min_epochs: Minimum epochs before early stopping can trigger
        early_stopping_patience: Stop if no improvement for this many epochs
        validation_split: Fraction of data for validation (0.0-1.0)
        gradient_clip: Gradient clipping threshold (0 = disabled)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """

    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    min_epochs: int = 10
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    gradient_clip: float = 0.0
    verbose: int = 1


# ============================================================================
# Callback System
# ============================================================================


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks allow customizing behavior during training without modifying
    the Trainer class. Override methods to add custom logic.
    """

    def set_trainer(self, trainer: "Trainer") -> None:
        """Set reference to the trainer (called automatically)."""
        self.trainer = trainer

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Called at the end of each epoch.

        Returns:
            False to stop training early, True to continue
        """
        return True


class EarlyStopping(Callback):
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        restore_best: Whether to restore best weights at end (not implemented)

    Example:
        >>> trainer.add_callback(EarlyStopping(patience=10, min_delta=0.001))
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best_loss = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        logs = logs or {}
        current_loss = logs.get("val_loss", logs.get("loss", None))

        if current_loss is None:
            return True

        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if hasattr(self, "trainer") and self.trainer.config.verbose > 0:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                return False

        return True


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        filepath: Path to save model (can include {epoch} placeholder)
        save_best_only: Only save when validation loss improves
        verbose: Print message when saving

    Example:
        >>> trainer.add_callback(ModelCheckpoint('model_epoch_{epoch}.kan'))
        >>> trainer.add_callback(ModelCheckpoint('best_model.kan', save_best_only=True))
    """

    def __init__(
        self,
        filepath: str,
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss: Optional[float] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best_loss = None

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        logs = logs or {}
        current_loss = logs.get("val_loss", logs.get("loss", None))

        if current_loss is None:
            return True

        should_save = False
        if self.save_best_only:
            if self.best_loss is None or current_loss < self.best_loss:
                self.best_loss = current_loss
                should_save = True
        else:
            should_save = True

        if should_save and hasattr(self, "trainer"):
            filepath = self.filepath.format(epoch=epoch + 1)
            self.trainer.model.save(filepath)
            if self.verbose:
                logger.info(f"Saved model to {filepath} (loss: {current_loss:.6f})")

        return True


class ProgressLogger(Callback):
    """Log training progress to console.

    Args:
        log_frequency: Print every N epochs (1 = every epoch)

    Example:
        >>> trainer.add_callback(ProgressLogger(log_frequency=10))
    """

    def __init__(self, log_frequency: int = 1):
        self.log_frequency = log_frequency
        self.epoch_start_time: float = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Training started...")
        self.train_start_time = time.time()

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        if (epoch + 1) % self.log_frequency != 0:
            return True

        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time

        parts = [f"Epoch {epoch + 1}"]

        if "loss" in logs:
            parts.append(f"loss: {logs['loss']:.6f}")
        if "val_loss" in logs:
            parts.append(f"val_loss: {logs['val_loss']:.6f}")

        parts.append(f"({elapsed:.2f}s)")

        logger.info(" - ".join(parts))
        return True

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        total_time = time.time() - self.train_start_time
        logger.info(f"Training completed in {total_time:.2f}s")


class LearningRateScheduler(Callback):
    """Adjust learning rate during training.

    Note: KANN uses RMSprop with fixed learning rate internally.
    This callback tracks what the LR *would* be for logging purposes,
    but cannot actually modify the internal optimizer.

    Args:
        schedule: Function (epoch, current_lr) -> new_lr
        verbose: Print when LR changes

    Example:
        >>> # Decay LR by 0.95 every 10 epochs
        >>> def schedule(epoch, lr):
        ...     if epoch > 0 and epoch % 10 == 0:
        ...         return lr * 0.95
        ...     return lr
        >>> trainer.add_callback(LearningRateScheduler(schedule))
    """

    def __init__(
        self,
        schedule: Callable[[int, float], float],
        verbose: bool = True,
    ):
        self.schedule = schedule
        self.verbose = verbose
        self.current_lr: float = 0.001

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if hasattr(self, "trainer"):
            self.current_lr = self.trainer.config.learning_rate

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        new_lr = self.schedule(epoch, self.current_lr)
        if new_lr != self.current_lr:
            if self.verbose:
                logger.info(f"Learning rate: {self.current_lr:.6f} -> {new_lr:.6f}")
            self.current_lr = new_lr


# ============================================================================
# Trainer Class
# ============================================================================


class Trainer:
    """High-level training interface for neural network models.

    Provides a convenient way to train models with callbacks, progress
    reporting, and training history tracking.

    Example:
        >>> config = TrainingConfig(learning_rate=0.001, max_epochs=100)
        >>> trainer = Trainer(model, config)
        >>> trainer.add_callback(EarlyStopping(patience=10))
        >>> history = trainer.train(x_train, y_train)
        >>> print(f"Best loss: {min(history['val_loss'])}")
    """

    def __init__(self, model: NeuralNetwork, config: Optional[TrainingConfig] = None):
        """Initialize trainer.

        Args:
            model: Neural network to train
            config: Training configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.callbacks: List[Callback] = []
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

    def add_callback(self, callback: Callback) -> "Trainer":
        """Add a callback.

        Args:
            callback: Callback instance to add

        Returns:
            Self for chaining
        """
        callback.set_trainer(self)
        self.callbacks.append(callback)
        return self

    def _notify_callbacks(
        self, method: str, *args, **kwargs
    ) -> bool:
        """Call a method on all callbacks.

        Returns:
            False if any callback returns False (for epoch_end), True otherwise
        """
        result = True
        for callback in self.callbacks:
            ret = getattr(callback, method)(*args, **kwargs)
            if method == "on_epoch_end" and ret is False:
                result = False
        return result

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            x_train: Training input data (numpy array or buffer)
            y_train: Training target data (numpy array or buffer)
            x_val: Validation input data (optional, uses validation_split if None)
            y_val: Validation target data (optional)

        Returns:
            Training history dict with 'loss' and 'val_loss' lists
        """
        # Reset history
        self.history = {"loss": [], "val_loss": []}

        # Notify callbacks
        self._notify_callbacks("on_train_begin", {"config": self.config})

        # If no validation data and validation_split > 0, split from training
        if x_val is None and self.config.validation_split > 0:
            split_idx = int(len(x_train) * (1 - self.config.validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]

        # Training loop - use KANN's built-in trainer for efficiency
        # KANN handles mini-batching, early stopping internally
        epochs_trained = self.model.train(
            x_train,
            y_train,
            learning_rate=self.config.learning_rate,
            mini_batch_size=self.config.batch_size,
            max_epochs=self.config.max_epochs,
            min_epochs=self.config.min_epochs,
            max_drop_streak=self.config.early_stopping_patience,
            validation_fraction=self.config.validation_split if x_val is None else 0.0,
        )

        # Compute final costs for history
        train_cost = self.model.cost(x_train, y_train)
        self.history["loss"].append(train_cost)

        if x_val is not None and len(x_val) > 0:
            val_cost = self.model.cost(x_val, y_val)
            self.history["val_loss"].append(val_cost)

        # Store epochs count in history for access by callers
        self.history["epochs_trained"] = epochs_trained

        logs = {
            "loss": train_cost,
            "epochs": epochs_trained,
        }
        if x_val is not None:
            logs["val_loss"] = val_cost

        # Notify callbacks
        self._notify_callbacks("on_train_end", logs)

        if self.config.verbose > 0:
            logger.info(f"Training completed: {epochs_trained} epochs, loss: {train_cost:.6f}")

        return self.history

    def train_epochs(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
    ) -> Dict[str, List[float]]:
        """Train the model epoch by epoch with callback support.

        This method provides finer control than train(), calling callbacks
        at each epoch boundary. However, it's less efficient as it doesn't
        use KANN's optimized internal training loop.

        Args:
            x_train: Training input data
            y_train: Training target data
            x_val: Validation input data (optional)
            y_val: Validation target data (optional)

        Returns:
            Training history dict with 'loss' and 'val_loss' lists
        """
        # Reset history
        self.history = {"loss": [], "val_loss": []}

        # Notify callbacks
        self._notify_callbacks("on_train_begin", {"config": self.config})

        # If no validation data provided, split from training
        if x_val is None and self.config.validation_split > 0:
            split_idx = int(len(x_train) * (1 - self.config.validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]

        # Epoch-by-epoch training
        for epoch in range(self.config.max_epochs):
            self._notify_callbacks("on_epoch_begin", epoch)

            # Train for one epoch
            self.model.train(
                x_train,
                y_train,
                learning_rate=self.config.learning_rate,
                mini_batch_size=self.config.batch_size,
                max_epochs=1,
                min_epochs=0,
                max_drop_streak=999,  # Don't use KANN's early stopping
                validation_fraction=0.0,
            )

            # Compute costs
            train_cost = self.model.cost(x_train, y_train)
            self.history["loss"].append(train_cost)

            logs = {"loss": train_cost, "epoch": epoch}

            if x_val is not None and len(x_val) > 0:
                val_cost = self.model.cost(x_val, y_val)
                self.history["val_loss"].append(val_cost)
                logs["val_loss"] = val_cost

            # Notify callbacks - check for early stop
            should_continue = self._notify_callbacks("on_epoch_end", epoch, logs)
            if not should_continue and epoch >= self.config.min_epochs:
                break

        # Notify end
        final_logs = {
            "loss": self.history["loss"][-1] if self.history["loss"] else 0,
            "epochs": len(self.history["loss"]),
        }
        if self.history["val_loss"]:
            final_logs["val_loss"] = self.history["val_loss"][-1]

        self._notify_callbacks("on_train_end", final_logs)

        return self.history

    def evaluate(self, x_test, y_test) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            x_test: Test input data
            y_test: Test target data

        Returns:
            Dict with 'loss' and other metrics
        """
        cost = self.model.cost(x_test, y_test)
        return {"loss": cost}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save model
        """
        self.model.save(path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load model from
        """
        self.model = NeuralNetwork.load(path)

    def train_rnn_sequences(
        self,
        sequences: List[List[int]],
        seq_length: int,
        vocab_size: int,
        grad_clip: float = 5.0,
    ) -> Dict[str, List[float]]:
        """Train an RNN model on token sequences using backpropagation through time.

        This method is specifically designed for RNN/LSTM/GRU models and uses
        proper BPTT (backpropagation through time) instead of the standard
        feedforward training API.

        Args:
            sequences: List of token sequences (each sequence is a list of ints)
            seq_length: Length of subsequences for BPTT unrolling
            vocab_size: Size of the vocabulary (number of possible tokens)
            grad_clip: Gradient clipping threshold (default 5.0)

        Returns:
            Training history dict with 'loss' and 'val_loss' lists

        Example:
            >>> # Prepare sequences (lists of token indices)
            >>> sequences = [[0, 1, 2, 3, 4, ...], [5, 6, 7, 8, ...], ...]
            >>> history = trainer.train_rnn_sequences(
            ...     sequences=sequences,
            ...     seq_length=32,
            ...     vocab_size=128,
            ... )
        """
        # Filter sequences that are too short
        valid_sequences = [s for s in sequences if len(s) > seq_length]
        if not valid_sequences:
            logger.warning("No sequences long enough for training")
            return {"loss": [], "val_loss": []}

        # Reset history
        self.history = {"loss": [], "val_loss": []}

        # Notify callbacks
        self._notify_callbacks("on_train_begin", {"config": self.config})

        # Use the model's train_rnn method
        history = self.model.train_rnn(
            sequences=valid_sequences,
            seq_length=seq_length,
            vocab_size=vocab_size,
            learning_rate=self.config.learning_rate,
            mini_batch_size=self.config.batch_size,
            max_epochs=self.config.max_epochs,
            grad_clip=grad_clip,
            validation_fraction=self.config.validation_split,
            verbose=self.config.verbose,
        )

        # Copy history from train_rnn result
        self.history["loss"] = history.get("loss", [])
        self.history["val_loss"] = history.get("val_loss", [])

        # Notify callbacks
        final_logs = {
            "loss": self.history["loss"][-1] if self.history["loss"] else 0,
            "epochs": len(self.history["loss"]),
        }
        if self.history["val_loss"]:
            final_logs["val_loss"] = self.history["val_loss"][-1]

        self._notify_callbacks("on_train_end", final_logs)

        if self.config.verbose > 0:
            logger.info(
                f"RNN Training completed: {len(self.history['loss'])} epochs, "
                f"final loss: {final_logs['loss']:.6f}"
            )

        return self.history
