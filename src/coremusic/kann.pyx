# kann.pyx - Cython wrapper for the KANN neural network library
# KANN: Klib Artificial Neural Network
# https://github.com/attractivechaos/kann
#
# This module provides Python bindings for training neural networks
# using the lightweight KANN library, particularly for MIDI generation.
#
# Design: Uses typed memoryviews and the buffer protocol for array handling,
# making it compatible with array.array, numpy.ndarray, or any buffer-compatible type.

cimport cython
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memcpy, memset

from .kann cimport *

import array

# ============================================================================
# Constants - Exposed to Python
# ============================================================================

# Node flags
KANN_FLAG_IN = KANN_F_IN
KANN_FLAG_OUT = KANN_F_OUT
KANN_FLAG_TRUTH = KANN_F_TRUTH
KANN_FLAG_COST = KANN_F_COST

# Cost function types
COST_BINARY_CROSS_ENTROPY = KANN_C_CEB
COST_MULTI_CROSS_ENTROPY = KANN_C_CEM
COST_BINARY_CROSS_ENTROPY_NEG = KANN_C_CEB_NEG
COST_MSE = KANN_C_MSE

# RNN flags
RNN_VAR_H0 = KANN_RNN_VAR_H0
RNN_NORM = KANN_RNN_NORM

# KAD flags
KAD_FLAG_VAR = KAD_VAR
KAD_FLAG_CONST = KAD_CONST


# ============================================================================
# Exception Classes
# ============================================================================

class KannError(Exception):
    """Base exception for KANN errors."""
    pass


class KannModelError(KannError):
    """Error related to model operations."""
    pass


class KannTrainingError(KannError):
    """Error during training."""
    pass


# ============================================================================
# Low-level Node Wrapper
# ============================================================================

cdef class KadNode:
    """Wrapper for a computational graph node (kad_node_t)."""
    cdef kad_node_t* _node
    cdef bint _owned

    def __cinit__(self):
        self._node = NULL
        self._owned = False

    def __dealloc__(self):
        # Nodes are typically managed by the graph/network
        pass

    @staticmethod
    cdef KadNode wrap(kad_node_t* node):
        """Wrap an existing C node pointer."""
        cdef KadNode wrapper = KadNode.__new__(KadNode)
        wrapper._node = node
        wrapper._owned = False
        return wrapper

    @property
    def n_dimensions(self):
        """Number of dimensions."""
        if self._node == NULL:
            return 0
        return self._node.n_d

    @property
    def dimensions(self):
        """Get dimensions as a tuple."""
        if self._node == NULL:
            return ()
        return tuple(self._node.d[i] for i in range(self._node.n_d))

    @property
    def n_children(self):
        """Number of child nodes."""
        if self._node == NULL:
            return 0
        return self._node.n_child

    @property
    def op(self):
        """Operator ID."""
        if self._node == NULL:
            return 0
        return self._node.op

    @property
    def flag(self):
        """Node flag."""
        if self._node == NULL:
            return 0
        return self._node.flag

    @property
    def ext_label(self):
        """External label."""
        if self._node == NULL:
            return 0
        return self._node.ext_label

    @property
    def ext_flag(self):
        """External flag."""
        if self._node == NULL:
            return 0
        return self._node.ext_flag

    def is_valid(self):
        """Check if this node wraps a valid pointer."""
        return self._node != NULL


# ============================================================================
# Neural Network Class
# ============================================================================

cdef class NeuralNetwork:
    """
    High-level wrapper for KANN neural networks.

    This class provides a Pythonic interface to create, train, and use
    neural networks with the KANN library.

    Example usage for MIDI generation:

        # Create a simple MLP for note prediction
        nn = NeuralNetwork.mlp(
            input_size=128,   # e.g., one-hot encoded note
            hidden_sizes=[256, 128],
            output_size=128,
            cost_type=COST_MULTI_CROSS_ENTROPY
        )

        # Train
        nn.train(x_train, y_train, learning_rate=0.001, epochs=100)

        # Generate
        output = nn.apply(input_vector)
    """
    cdef kann_t* _ann
    cdef bint _is_unrolled
    cdef int _input_dim
    cdef int _output_dim

    def __cinit__(self):
        self._ann = NULL
        self._is_unrolled = False
        self._input_dim = 0
        self._output_dim = 0

    def __dealloc__(self):
        if self._ann != NULL:
            if self._is_unrolled:
                kann_delete_unrolled(self._ann)
            else:
                kann_delete(self._ann)
            self._ann = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly release resources."""
        if self._ann != NULL:
            if self._is_unrolled:
                kann_delete_unrolled(self._ann)
            else:
                kann_delete(self._ann)
            self._ann = NULL

    @staticmethod
    cdef NeuralNetwork _wrap(kann_t* ann, bint is_unrolled=False):
        """Wrap an existing KANN network."""
        cdef NeuralNetwork wrapper = NeuralNetwork.__new__(NeuralNetwork)
        wrapper._ann = ann
        wrapper._is_unrolled = is_unrolled
        if ann != NULL:
            wrapper._input_dim = kann_feed_dim(ann, KANN_F_IN, 0)
            wrapper._output_dim = kann_feed_dim(ann, KANN_F_TRUTH, 0)
        return wrapper

    @staticmethod
    def mlp(int input_size, list hidden_sizes, int output_size,
            int cost_type=KANN_C_CEM, float dropout=0.0):
        """
        Create a multi-layer perceptron (feedforward network).

        Args:
            input_size: Size of input vector
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output vector
            cost_type: Loss function (COST_MULTI_CROSS_ENTROPY, COST_MSE, etc.)
            dropout: Dropout rate (0.0 = no dropout)

        Returns:
            NeuralNetwork instance
        """
        cdef kad_node_t* t
        cdef kann_t* ann

        # Create input layer
        t = kann_layer_input(input_size)
        if t == NULL:
            raise KannModelError("Failed to create input layer")

        # Create hidden layers
        for size in hidden_sizes:
            t = kann_layer_dense(t, size)
            if t == NULL:
                raise KannModelError("Failed to create dense layer")
            t = kad_relu(t)
            if t == NULL:
                raise KannModelError("Failed to create ReLU activation")
            if dropout > 0.0:
                t = kann_layer_dropout(t, dropout)
                if t == NULL:
                    raise KannModelError("Failed to create dropout layer")

        # Create output/cost layer
        t = kann_layer_cost(t, output_size, cost_type)
        if t == NULL:
            raise KannModelError("Failed to create cost layer")

        # Create network
        ann = kann_new(t, 0)
        if ann == NULL:
            raise KannModelError("Failed to create neural network")

        return NeuralNetwork._wrap(ann)

    @staticmethod
    def lstm(int input_size, int hidden_size, int output_size,
             int cost_type=KANN_C_CEM, int rnn_flags=0):
        """
        Create an LSTM network for sequence modeling.

        Args:
            input_size: Size of input at each timestep
            hidden_size: Size of LSTM hidden state
            output_size: Size of output vector
            cost_type: Loss function
            rnn_flags: RNN configuration flags

        Returns:
            NeuralNetwork instance
        """
        cdef kad_node_t* t
        cdef kann_t* ann

        t = kann_layer_input(input_size)
        if t == NULL:
            raise KannModelError("Failed to create input layer")

        t = kann_layer_lstm(t, hidden_size, rnn_flags)
        if t == NULL:
            raise KannModelError("Failed to create LSTM layer")

        t = kann_layer_cost(t, output_size, cost_type)
        if t == NULL:
            raise KannModelError("Failed to create cost layer")

        ann = kann_new(t, 0)
        if ann == NULL:
            raise KannModelError("Failed to create neural network")

        return NeuralNetwork._wrap(ann)

    @staticmethod
    def gru(int input_size, int hidden_size, int output_size,
            int cost_type=KANN_C_CEM, int rnn_flags=0):
        """
        Create a GRU network for sequence modeling.

        Args:
            input_size: Size of input at each timestep
            hidden_size: Size of GRU hidden state
            output_size: Size of output vector
            cost_type: Loss function
            rnn_flags: RNN configuration flags

        Returns:
            NeuralNetwork instance
        """
        cdef kad_node_t* t
        cdef kann_t* ann

        t = kann_layer_input(input_size)
        if t == NULL:
            raise KannModelError("Failed to create input layer")

        t = kann_layer_gru(t, hidden_size, rnn_flags)
        if t == NULL:
            raise KannModelError("Failed to create GRU layer")

        t = kann_layer_cost(t, output_size, cost_type)
        if t == NULL:
            raise KannModelError("Failed to create cost layer")

        ann = kann_new(t, 0)
        if ann == NULL:
            raise KannModelError("Failed to create neural network")

        return NeuralNetwork._wrap(ann)

    @staticmethod
    def rnn(int input_size, int hidden_size, int output_size,
            int cost_type=KANN_C_CEM, int rnn_flags=0):
        """
        Create a simple RNN network.

        Args:
            input_size: Size of input at each timestep
            hidden_size: Size of RNN hidden state
            output_size: Size of output vector
            cost_type: Loss function
            rnn_flags: RNN configuration flags

        Returns:
            NeuralNetwork instance
        """
        cdef kad_node_t* t
        cdef kann_t* ann

        t = kann_layer_input(input_size)
        if t == NULL:
            raise KannModelError("Failed to create input layer")

        t = kann_layer_rnn(t, hidden_size, rnn_flags)
        if t == NULL:
            raise KannModelError("Failed to create RNN layer")

        t = kann_layer_cost(t, output_size, cost_type)
        if t == NULL:
            raise KannModelError("Failed to create cost layer")

        ann = kann_new(t, 0)
        if ann == NULL:
            raise KannModelError("Failed to create neural network")

        return NeuralNetwork._wrap(ann)

    @staticmethod
    def load(str filename):
        """
        Load a network from file.

        Args:
            filename: Path to model file

        Returns:
            NeuralNetwork instance
        """
        import os
        if not os.path.exists(filename):
            raise KannModelError(f"Model file not found: {filename}")
        cdef bytes fn_bytes = filename.encode('utf-8')
        cdef kann_t* ann = kann_load(fn_bytes)
        if ann == NULL:
            raise KannModelError(f"Failed to load model from {filename}")
        return NeuralNetwork._wrap(ann)

    def save(self, str filename):
        """
        Save the network to file.

        Args:
            filename: Path to save model
        """
        if self._ann == NULL:
            raise KannModelError("No network to save")
        cdef bytes fn_bytes = filename.encode('utf-8')
        kann_save(fn_bytes, self._ann)

    @property
    def n_nodes(self):
        """Number of nodes in the computational graph."""
        if self._ann == NULL:
            return 0
        return self._ann.n

    @property
    def input_dim(self):
        """Input dimension."""
        return self._input_dim

    @property
    def output_dim(self):
        """Output dimension."""
        return self._output_dim

    @property
    def n_var(self):
        """Total number of trainable variables."""
        if self._ann == NULL:
            return 0
        return kad_size_var(self._ann.n, self._ann.v)

    @property
    def n_const(self):
        """Total number of constants."""
        if self._ann == NULL:
            return 0
        return kad_size_const(self._ann.n, self._ann.v)

    def set_batch_size(self, int batch_size):
        """Set the mini-batch size."""
        if self._ann == NULL:
            raise KannModelError("No network")
        kad_sync_dim(self._ann.n, self._ann.v, batch_size)

    def switch_mode(self, bint is_training):
        """Switch between training and inference mode."""
        if self._ann == NULL:
            raise KannModelError("No network")
        kann_switch(self._ann, 1 if is_training else 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(self, float[:, :] x, float[:, :] y,
              float learning_rate=0.001,
              int mini_batch_size=64,
              int max_epochs=100,
              int min_epochs=0,
              int max_drop_streak=10,
              float validation_fraction=0.1):
        """
        Train the network using the built-in feedforward trainer.

        Args:
            x: Input data, shape (n_samples, input_dim) - any buffer with float32 data
            y: Target data, shape (n_samples, output_dim) - any buffer with float32 data
            learning_rate: Learning rate for RMSprop
            mini_batch_size: Size of mini-batches
            max_epochs: Maximum number of epochs
            min_epochs: Minimum number of epochs before early stopping
            max_drop_streak: Stop if validation loss doesn't improve for this many epochs
            validation_fraction: Fraction of data to use for validation

        Returns:
            Number of epochs trained

        Note:
            Accepts any buffer-protocol compatible object (numpy.ndarray, array.array, etc.)
        """
        if self._ann == NULL:
            raise KannModelError("No network to train")

        cdef int n_samples = x.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("x and y must have same number of samples")

        # Allocate array of pointers
        cdef float** x_ptrs = <float**>malloc(n_samples * sizeof(float*))
        cdef float** y_ptrs = <float**>malloc(n_samples * sizeof(float*))
        if x_ptrs == NULL or y_ptrs == NULL:
            if x_ptrs != NULL:
                free(x_ptrs)
            if y_ptrs != NULL:
                free(y_ptrs)
            raise MemoryError("Failed to allocate memory for training data")

        cdef int i
        for i in range(n_samples):
            x_ptrs[i] = &x[i, 0]
            y_ptrs[i] = &y[i, 0]

        cdef int result
        try:
            if min_epochs > 0:
                result = kann_train_fnn1b(
                    self._ann, learning_rate, mini_batch_size,
                    max_epochs, min_epochs, max_drop_streak,
                    validation_fraction, n_samples, x_ptrs, y_ptrs
                )
            else:
                result = kann_train_fnn1(
                    self._ann, learning_rate, mini_batch_size,
                    max_epochs, max_drop_streak,
                    validation_fraction, n_samples, x_ptrs, y_ptrs
                )
        finally:
            free(x_ptrs)
            free(y_ptrs)

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train_rnn(self, sequences, int seq_length, int vocab_size,
                  float learning_rate=0.001,
                  int mini_batch_size=32,
                  int max_epochs=100,
                  float grad_clip=10.0,
                  float validation_fraction=0.1,
                  int verbose=1):
        """
        Train an RNN using proper backpropagation through time (BPTT).

        This method unrolls the RNN and trains it on sequences, which is
        required for LSTM, GRU, and RNN models to learn effectively.

        Args:
            sequences: List of integer sequences (token indices)
            seq_length: Length of unrolled sequence for BPTT
            vocab_size: Size of vocabulary (for one-hot encoding)
            learning_rate: Learning rate for RMSprop
            mini_batch_size: Size of mini-batches
            max_epochs: Maximum number of epochs
            grad_clip: Gradient clipping threshold (0 = disabled)
            validation_fraction: Fraction of data for validation
            verbose: Verbosity level (0=silent, 1=progress)

        Returns:
            Dictionary with training history: {'loss': [...], 'val_loss': [...]}
        """
        import random

        if self._ann == NULL:
            raise KannModelError("No network to train")

        cdef int n_var = kad_size_var(self._ann.n, self._ann.v)
        cdef int n_seq = len(sequences)
        cdef int n_train, n_val
        cdef int epoch, batch_idx, t, b, node_idx, node_len
        cdef int seq_idx, seq_len
        cdef float total_cost, batch_cost, avg_loss, avg_val_loss
        cdef int actual_batch_size, tokens_processed
        cdef float clipped, val_cost
        cdef int val_tokens, start_pos, token, next_token
        cdef int val_batch_idx, val_actual_batch, valid_in_batch
        cdef size_t buf_size
        cdef kad_node_t* node

        # Split into training and validation
        n_val = int(n_seq * validation_fraction)
        n_train = n_seq - n_val

        if n_train < mini_batch_size:
            mini_batch_size = n_train if n_train > 0 else 1

        # Unroll the RNN for seq_length time steps
        cdef kann_t* ua = NULL
        cdef int* len_arr = <int*>malloc(sizeof(int))
        if len_arr == NULL:
            raise MemoryError("Failed to allocate memory")
        len_arr[0] = seq_length

        try:
            ua = kann_unroll_array(self._ann, len_arr)
        finally:
            free(len_arr)

        if ua == NULL:
            raise KannModelError("Failed to unroll network (is it an RNN?)")

        # Allocate arrays for input/output at each time step
        cdef float** x_seq = <float**>malloc(seq_length * sizeof(float*))
        cdef float** y_seq = <float**>malloc(seq_length * sizeof(float*))
        cdef float* r = <float*>calloc(n_var, sizeof(float))  # RMSprop memory

        if x_seq == NULL or y_seq == NULL or r == NULL:
            if x_seq != NULL:
                free(x_seq)
            if y_seq != NULL:
                free(y_seq)
            if r != NULL:
                free(r)
            kann_delete_unrolled(ua)
            raise MemoryError("Failed to allocate memory for training")

        # Initialize all pointers to NULL first
        for t in range(seq_length):
            x_seq[t] = NULL
            y_seq[t] = NULL

        # Allocate data buffers for each time step
        buf_size = vocab_size * mini_batch_size
        for t in range(seq_length):
            x_seq[t] = <float*>calloc(buf_size, sizeof(float))
            y_seq[t] = <float*>calloc(buf_size, sizeof(float))
            if x_seq[t] == NULL or y_seq[t] == NULL:
                # Clean up on failure
                for b in range(seq_length):
                    if x_seq[b] != NULL:
                        free(x_seq[b])
                    if y_seq[b] != NULL:
                        free(y_seq[b])
                free(x_seq)
                free(y_seq)
                free(r)
                kann_delete_unrolled(ua)
                raise MemoryError("Failed to allocate time step buffers")

        # Training history
        history = {'loss': [], 'val_loss': []}

        # Set batch size and bind inputs/outputs
        kad_sync_dim(ua.n, ua.v, mini_batch_size)
        kann_switch(ua, 1)  # Training mode
        kann_feed_bind(ua, KANN_F_IN, 0, x_seq)
        kann_feed_bind(ua, KANN_F_TRUTH, 0, y_seq)

        try:
            for epoch in range(max_epochs):
                total_cost = 0.0
                tokens_processed = 0

                # Shuffle training sequences
                train_indices = list(range(n_train))
                random.shuffle(train_indices)

                # Process sequences in mini-batches
                batch_idx = 0
                while batch_idx < n_train:
                    actual_batch_size = min(mini_batch_size, n_train - batch_idx)

                    # Reset hidden states for all nodes with pre pointer
                    node_idx = 0
                    while node_idx < ua.n:
                        node = ua.v[node_idx]
                        if node.pre != NULL and node.x != NULL:
                            node_len = kad_len(node)
                            if node_len > 0:
                                memset(node.x, 0, node_len * sizeof(float))
                        node_idx += 1

                    # Clear input/output buffers
                    for t in range(seq_length):
                        memset(x_seq[t], 0, buf_size * sizeof(float))
                        memset(y_seq[t], 0, buf_size * sizeof(float))

                    # Fill in the batch
                    for b in range(actual_batch_size):
                        seq_idx = train_indices[batch_idx + b]
                        seq = sequences[seq_idx]
                        seq_len = len(seq)

                        if seq_len <= seq_length:
                            continue

                        # Random starting position
                        start_pos = random.randint(0, seq_len - seq_length - 1)

                        for t in range(seq_length):
                            # Input: current token
                            token = seq[start_pos + t]
                            if 0 <= token < vocab_size:
                                x_seq[t][b * vocab_size + token] = 1.0

                            # Target: next token
                            next_token = seq[start_pos + t + 1]
                            if 0 <= next_token < vocab_size:
                                y_seq[t][b * vocab_size + next_token] = 1.0

                    # Forward and backward pass
                    batch_cost = kann_cost(ua, 0, 1)  # 1 = compute gradients
                    total_cost += batch_cost * seq_length * actual_batch_size
                    tokens_processed += seq_length * actual_batch_size

                    # Gradient clipping
                    if grad_clip > 0.0:
                        clipped = kann_grad_clip(grad_clip, n_var, ua.g)

                    # RMSprop update
                    kann_RMSprop(n_var, learning_rate, NULL, 0.9, ua.g, ua.x, r)

                    batch_idx += mini_batch_size

                # Compute average loss
                avg_loss = total_cost / tokens_processed if tokens_processed > 0 else 0.0
                history['loss'].append(avg_loss)

                # Validation (compute cost on held-out sequences)
                # Must match training: accumulate cost * seq_length * batch_size, divide by total tokens
                if n_val > 0:
                    val_cost = 0.0
                    val_tokens = 0
                    kann_switch(ua, 0)  # Inference mode

                    # Process validation sequences in mini-batches (like training)
                    val_batch_idx = 0
                    while val_batch_idx < n_val:
                        val_actual_batch = min(mini_batch_size, n_val - val_batch_idx)

                        # Reset hidden states
                        node_idx = 0
                        while node_idx < ua.n:
                            node = ua.v[node_idx]
                            if node.pre != NULL and node.x != NULL:
                                node_len = kad_len(node)
                                if node_len > 0:
                                    memset(node.x, 0, node_len * sizeof(float))
                            node_idx += 1

                        # Clear buffers
                        t = 0
                        while t < seq_length:
                            memset(x_seq[t], 0, buf_size * sizeof(float))
                            memset(y_seq[t], 0, buf_size * sizeof(float))
                            t += 1

                        # Fill in the validation batch
                        valid_in_batch = 0
                        b = 0
                        while b < val_actual_batch:
                            seq_idx = n_train + val_batch_idx + b
                            seq = sequences[seq_idx]
                            seq_len = len(seq)
                            b += 1

                            if seq_len <= seq_length:
                                continue

                            # Use a fixed position for validation (deterministic)
                            start_pos = 0

                            t = 0
                            while t < seq_length:
                                token = seq[start_pos + t]
                                if 0 <= token < vocab_size:
                                    x_seq[t][valid_in_batch * vocab_size + token] = 1.0
                                next_token = seq[start_pos + t + 1]
                                if 0 <= next_token < vocab_size:
                                    y_seq[t][valid_in_batch * vocab_size + next_token] = 1.0
                                t += 1

                            valid_in_batch += 1

                        if valid_in_batch > 0:
                            # Accumulate cost same as training: cost * seq_length * batch_size
                            val_cost += kann_cost(ua, 0, 0) * seq_length * valid_in_batch
                            val_tokens += seq_length * valid_in_batch

                        val_batch_idx += val_actual_batch

                    avg_val_loss = val_cost / val_tokens if val_tokens > 0 else 0.0
                    history['val_loss'].append(avg_val_loss)
                    kann_switch(ua, 1)  # Back to training mode

                if verbose > 0:
                    if n_val > 0:
                        print(f"epoch: {epoch + 1}; train cost: {avg_loss:.4f}; val cost: {avg_val_loss:.4f}")
                    else:
                        print(f"epoch: {epoch + 1}; train cost: {avg_loss:.4f}")

        finally:
            # Clean up
            for t in range(seq_length):
                if x_seq[t] != NULL:
                    free(x_seq[t])
                if y_seq[t] != NULL:
                    free(y_seq[t])
            free(x_seq)
            free(y_seq)
            free(r)
            kann_delete_unrolled(ua)

        return history

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def apply(self, float[:] x):
        """
        Apply the network to a single input.

        Args:
            x: Input vector - any buffer with float32 data

        Returns:
            Output vector as array.array('f', ...)

        Note:
            Accepts any buffer-protocol compatible object (numpy.ndarray, array.array, etc.)
        """
        if self._ann == NULL:
            raise KannModelError("No network")

        cdef const float* result = kann_apply1(self._ann, &x[0])
        if result == NULL:
            raise KannModelError("Forward pass failed")

        # Copy result to array.array (works without numpy)
        cdef int out_dim = self._output_dim if self._output_dim > 0 else kann_feed_dim(self._ann, KANN_F_OUT, 0)
        output = array.array('f', [0.0] * out_dim)
        cdef float[:] output_view = output
        memcpy(&output_view[0], result, out_dim * sizeof(float))
        return output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cost(self, float[:, :] x, float[:, :] y):
        """
        Compute the cost over a dataset.

        Args:
            x: Input data - any buffer with float32 data
            y: Target data - any buffer with float32 data

        Returns:
            Average cost

        Note:
            Accepts any buffer-protocol compatible object (numpy.ndarray, array.array, etc.)
        """
        if self._ann == NULL:
            raise KannModelError("No network")

        cdef int n_samples = x.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("x and y must have same number of samples")

        cdef float** x_ptrs = <float**>malloc(n_samples * sizeof(float*))
        cdef float** y_ptrs = <float**>malloc(n_samples * sizeof(float*))
        if x_ptrs == NULL or y_ptrs == NULL:
            if x_ptrs != NULL:
                free(x_ptrs)
            if y_ptrs != NULL:
                free(y_ptrs)
            raise MemoryError("Failed to allocate memory")

        cdef int i
        for i in range(n_samples):
            x_ptrs[i] = &x[i, 0]
            y_ptrs[i] = &y[i, 0]

        cdef float result
        try:
            result = kann_cost_fnn1(self._ann, n_samples, x_ptrs, y_ptrs)
        finally:
            free(x_ptrs)
            free(y_ptrs)

        return result

    def clone(self, int batch_size=1):
        """
        Clone the network with a different batch size.

        Args:
            batch_size: New batch size

        Returns:
            New NeuralNetwork instance
        """
        if self._ann == NULL:
            raise KannModelError("No network to clone")
        cdef kann_t* cloned = kann_clone(self._ann, batch_size)
        if cloned == NULL:
            raise KannModelError("Failed to clone network")
        return NeuralNetwork._wrap(cloned)

    def unroll(self, int length):
        """
        Unroll an RNN for a specified number of time steps.

        Args:
            length: Number of time steps to unroll

        Returns:
            New unrolled NeuralNetwork instance
        """
        if self._ann == NULL:
            raise KannModelError("No network to unroll")

        cdef int* len_arr = <int*>malloc(sizeof(int))
        if len_arr == NULL:
            raise MemoryError("Failed to allocate memory")
        len_arr[0] = length

        cdef kann_t* unrolled
        try:
            unrolled = kann_unroll_array(self._ann, len_arr)
        finally:
            free(len_arr)

        if unrolled == NULL:
            raise KannModelError("Failed to unroll network (is it an RNN?)")
        return NeuralNetwork._wrap(unrolled, is_unrolled=True)

    def rnn_start(self):
        """Start RNN continuous feeding mode."""
        if self._ann == NULL:
            raise KannModelError("No network")
        kann_rnn_start(self._ann)

    def rnn_end(self):
        """End RNN continuous feeding mode."""
        if self._ann == NULL:
            raise KannModelError("No network")
        kann_rnn_end(self._ann)

    def enable_multithreading(self, int n_threads, int max_batch_size):
        """
        Enable multi-threaded training.

        Args:
            n_threads: Number of threads
            max_batch_size: Maximum batch size
        """
        if self._ann == NULL:
            raise KannModelError("No network")
        kann_mt(self._ann, n_threads, max_batch_size)


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(unsigned long long seed):
    """Set the random seed for reproducibility."""
    kad_srand(NULL, seed)


def set_verbose(int level):
    """Set verbosity level for KANN operations."""
    global kann_verbose
    kann_verbose = level


# ============================================================================
# MIDI Generation Helpers
# ============================================================================

def one_hot_encode(int[:] values, int num_classes):
    """
    One-hot encode an array of integer values.

    Args:
        values: Array of integer class indices (any buffer with int32 data)
        num_classes: Total number of classes

    Returns:
        One-hot encoded list of lists, shape (len(values), num_classes)
    """
    cdef int n = values.shape[0]
    cdef int i, v

    # Create output as list of array.array for efficiency
    result = []
    for i in range(n):
        row = array.array('f', [0.0] * num_classes)
        v = values[i]
        if 0 <= v < num_classes:
            row[v] = 1.0
        result.append(row)
    return result


def one_hot_encode_2d(int[:] values, int num_classes):
    """
    One-hot encode an array of integer values, returning a flat 2D buffer.

    Args:
        values: Array of integer class indices (any buffer with int32 data)
        num_classes: Total number of classes

    Returns:
        Flat array.array('f', ...) with shape semantically (len(values), num_classes)
    """
    cdef int n = values.shape[0]
    cdef int i, v, offset

    # Create flat output array
    output = array.array('f', [0.0] * (n * num_classes))
    cdef float[:] out_view = output

    for i in range(n):
        v = values[i]
        if 0 <= v < num_classes:
            offset = i * num_classes + v
            out_view[offset] = 1.0
    return output


import math
import random


def softmax_sample(float[:] probs, float temperature=1.0):
    """
    Sample from a probability distribution with temperature scaling.

    Args:
        probs: Probability distribution (any buffer with float32 data)
        temperature: Temperature for sampling (higher = more random)

    Returns:
        Sampled index
    """
    cdef int n = probs.shape[0]
    cdef int i
    cdef double total = 0.0
    cdef double r, cumsum

    # Compute scaled probabilities
    scaled = array.array('d', [0.0] * n)
    cdef double[:] scaled_view = scaled

    if temperature != 1.0:
        for i in range(n):
            scaled_view[i] = math.exp(math.log(probs[i] + 1e-10) / temperature)
            total += scaled_view[i]
    else:
        for i in range(n):
            scaled_view[i] = probs[i]
            total += scaled_view[i]

    # Normalize
    for i in range(n):
        scaled_view[i] /= total

    # Sample
    r = random.random()
    cumsum = 0.0
    for i in range(n):
        cumsum += scaled_view[i]
        if r < cumsum:
            return i
    return n - 1


def prepare_sequence_data(sequences, int seq_length, int vocab_size):
    """
    Prepare sequence data for RNN training.

    Args:
        sequences: List of integer sequences
        seq_length: Length of each training sequence
        vocab_size: Size of vocabulary (e.g., 128 for MIDI notes)

    Returns:
        Tuple of (x_train, y_train) as lists of array.array

    Note:
        Returns list format. Convert to numpy or other format as needed for training.
    """
    x_list = []
    y_list = []

    cdef int i, j, val
    cdef int flat_size = seq_length * vocab_size

    for seq in sequences:
        if len(seq) <= seq_length:
            continue
        for i in range(len(seq) - seq_length):
            # Input: one-hot encoded sequence (flattened)
            x_seq = array.array('f', [0.0] * flat_size)
            for j in range(seq_length):
                val = seq[i + j]
                if 0 <= val < vocab_size:
                    x_seq[j * vocab_size + val] = 1.0
            x_list.append(x_seq)

            # Target: next token (one-hot)
            y_seq = array.array('f', [0.0] * vocab_size)
            val = seq[i + seq_length]
            if 0 <= val < vocab_size:
                y_seq[val] = 1.0
            y_list.append(y_seq)

    if not x_list:
        raise ValueError("No valid sequences found")

    return x_list, y_list


def list_to_2d_array(list_of_arrays):
    """
    Convert a list of 1D arrays to a single flat array suitable for 2D memoryview.

    Args:
        list_of_arrays: List of array.array or similar buffers, all same length

    Returns:
        Tuple of (flat_array, n_rows, n_cols) where flat_array is array.array('f', ...)
    """
    if not list_of_arrays:
        raise ValueError("Empty list")

    n_rows = len(list_of_arrays)
    n_cols = len(list_of_arrays[0])

    flat = array.array('f', [0.0] * (n_rows * n_cols))
    cdef float[:] flat_view = flat
    cdef int i, j, offset

    for i in range(n_rows):
        row = list_of_arrays[i]
        offset = i * n_cols
        for j in range(n_cols):
            flat_view[offset + j] = row[j]

    return flat, n_rows, n_cols


class Array2D:
    """
    Simple 2D array wrapper around a flat array.array for use with memoryviews.

    Example:
        arr = Array2D(100, 10)  # 100 rows, 10 columns
        arr[5, 3] = 1.0
        view = arr.as_memoryview()  # Use with kann functions
    """
    def __init__(self, int rows, int cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self._data = array.array('f', data)
        else:
            self._data = array.array('f', [0.0] * (rows * cols))

    def __getitem__(self, key):
        i, j = key
        return self._data[i * self.cols + j]

    def __setitem__(self, key, value):
        i, j = key
        self._data[i * self.cols + j] = value

    def as_memoryview(self):
        """Return a 2D memoryview of the data."""
        cdef float[:] flat = self._data
        return flat.cast('f', [self.rows, self.cols]) if hasattr(flat, 'cast') else self._data

    @property
    def data(self):
        """Return the underlying flat array."""
        return self._data

    @classmethod
    def from_list(cls, list_of_lists):
        """Create Array2D from list of lists."""
        rows = len(list_of_lists)
        cols = len(list_of_lists[0]) if rows > 0 else 0
        flat = []
        for row in list_of_lists:
            flat.extend(row)
        return cls(rows, cols, flat)


# ============================================================================
# Graph Builder (for advanced usage)
# ============================================================================

cdef class GraphBuilder:
    """
    Low-level graph builder for creating custom network architectures.

    Example:
        builder = GraphBuilder()
        x = builder.input(128)
        h = builder.dense(x, 256)
        h = builder.relu(h)
        h = builder.dense(h, 128)
        cost = builder.softmax_cross_entropy(h, 128)
        nn = builder.build(cost)
    """
    cdef list _nodes

    def __cinit__(self):
        self._nodes = []

    def input(self, int size):
        """Create an input layer."""
        cdef kad_node_t* node = kann_layer_input(size)
        if node == NULL:
            raise KannModelError("Failed to create input layer")
        return KadNode.wrap(node)

    def dense(self, KadNode inp, int output_size):
        """Create a dense (fully connected) layer."""
        cdef kad_node_t* node = kann_layer_dense(inp._node, output_size)
        if node == NULL:
            raise KannModelError("Failed to create dense layer")
        return KadNode.wrap(node)

    def dropout(self, KadNode inp, float rate):
        """Create a dropout layer."""
        cdef kad_node_t* node = kann_layer_dropout(inp._node, rate)
        if node == NULL:
            raise KannModelError("Failed to create dropout layer")
        return KadNode.wrap(node)

    def layernorm(self, KadNode inp):
        """Create a layer normalization layer."""
        cdef kad_node_t* node = kann_layer_layernorm(inp._node)
        if node == NULL:
            raise KannModelError("Failed to create layernorm layer")
        return KadNode.wrap(node)

    def relu(self, KadNode inp):
        """Apply ReLU activation."""
        cdef kad_node_t* node = kad_relu(inp._node)
        if node == NULL:
            raise KannModelError("Failed to create ReLU")
        return KadNode.wrap(node)

    def sigmoid(self, KadNode inp):
        """Apply sigmoid activation."""
        cdef kad_node_t* node = kad_sigm(inp._node)
        if node == NULL:
            raise KannModelError("Failed to create sigmoid")
        return KadNode.wrap(node)

    def tanh(self, KadNode inp):
        """Apply tanh activation."""
        cdef kad_node_t* node = kad_tanh(inp._node)
        if node == NULL:
            raise KannModelError("Failed to create tanh")
        return KadNode.wrap(node)

    def softmax(self, KadNode inp):
        """Apply softmax activation."""
        cdef kad_node_t* node = kad_softmax(inp._node)
        if node == NULL:
            raise KannModelError("Failed to create softmax")
        return KadNode.wrap(node)

    def lstm(self, KadNode inp, int hidden_size, int flags=0):
        """Create an LSTM layer."""
        cdef kad_node_t* node = kann_layer_lstm(inp._node, hidden_size, flags)
        if node == NULL:
            raise KannModelError("Failed to create LSTM layer")
        return KadNode.wrap(node)

    def gru(self, KadNode inp, int hidden_size, int flags=0):
        """Create a GRU layer."""
        cdef kad_node_t* node = kann_layer_gru(inp._node, hidden_size, flags)
        if node == NULL:
            raise KannModelError("Failed to create GRU layer")
        return KadNode.wrap(node)

    def rnn(self, KadNode inp, int hidden_size, int flags=0):
        """Create a simple RNN layer."""
        cdef kad_node_t* node = kann_layer_rnn(inp._node, hidden_size, flags)
        if node == NULL:
            raise KannModelError("Failed to create RNN layer")
        return KadNode.wrap(node)

    def conv1d(self, KadNode inp, int n_filters, int kernel_size,
               int stride=1, int pad=0):
        """Create a 1D convolution layer."""
        cdef kad_node_t* node = kann_layer_conv1d(
            inp._node, n_filters, kernel_size, stride, pad
        )
        if node == NULL:
            raise KannModelError("Failed to create conv1d layer")
        return KadNode.wrap(node)

    def conv2d(self, KadNode inp, int n_filters, int k_rows, int k_cols,
               int stride_r=1, int stride_c=1, int pad_r=0, int pad_c=0):
        """Create a 2D convolution layer."""
        cdef kad_node_t* node = kann_layer_conv2d(
            inp._node, n_filters, k_rows, k_cols,
            stride_r, stride_c, pad_r, pad_c
        )
        if node == NULL:
            raise KannModelError("Failed to create conv2d layer")
        return KadNode.wrap(node)

    def add(self, KadNode x, KadNode y):
        """Element-wise addition."""
        cdef kad_node_t* node = kad_add(x._node, y._node)
        if node == NULL:
            raise KannModelError("Failed to create add operation")
        return KadNode.wrap(node)

    def sub(self, KadNode x, KadNode y):
        """Element-wise subtraction."""
        cdef kad_node_t* node = kad_sub(x._node, y._node)
        if node == NULL:
            raise KannModelError("Failed to create sub operation")
        return KadNode.wrap(node)

    def mul(self, KadNode x, KadNode y):
        """Element-wise multiplication."""
        cdef kad_node_t* node = kad_mul(x._node, y._node)
        if node == NULL:
            raise KannModelError("Failed to create mul operation")
        return KadNode.wrap(node)

    def matmul(self, KadNode x, KadNode y):
        """Matrix multiplication."""
        cdef kad_node_t* node = kad_matmul(x._node, y._node)
        if node == NULL:
            raise KannModelError("Failed to create matmul operation")
        return KadNode.wrap(node)

    def mse_cost(self, KadNode pred, KadNode truth):
        """Mean squared error cost."""
        cdef kad_node_t* node = kad_mse(pred._node, truth._node)
        if node == NULL:
            raise KannModelError("Failed to create MSE cost")
        return KadNode.wrap(node)

    def binary_cross_entropy(self, KadNode pred, KadNode truth):
        """Binary cross-entropy cost."""
        cdef kad_node_t* node = kad_ce_bin(pred._node, truth._node)
        if node == NULL:
            raise KannModelError("Failed to create BCE cost")
        return KadNode.wrap(node)

    def multi_cross_entropy(self, KadNode pred, KadNode truth):
        """Multi-class cross-entropy cost."""
        cdef kad_node_t* node = kad_ce_multi(pred._node, truth._node)
        if node == NULL:
            raise KannModelError("Failed to create CE cost")
        return KadNode.wrap(node)

    def softmax_cross_entropy(self, KadNode inp, int n_out):
        """
        Create a softmax + cross-entropy cost layer.
        This is equivalent to kann_layer_cost with KANN_C_CEM.
        """
        cdef kad_node_t* node = kann_layer_cost(inp._node, n_out, KANN_C_CEM)
        if node == NULL:
            raise KannModelError("Failed to create softmax CE cost")
        return KadNode.wrap(node)

    def sigmoid_cross_entropy(self, KadNode inp, int n_out):
        """
        Create a sigmoid + binary cross-entropy cost layer.
        This is equivalent to kann_layer_cost with KANN_C_CEB.
        """
        cdef kad_node_t* node = kann_layer_cost(inp._node, n_out, KANN_C_CEB)
        if node == NULL:
            raise KannModelError("Failed to create sigmoid CE cost")
        return KadNode.wrap(node)

    def mse_layer(self, KadNode inp, int n_out):
        """
        Create an MSE cost layer.
        This is equivalent to kann_layer_cost with KANN_C_MSE.
        """
        cdef kad_node_t* node = kann_layer_cost(inp._node, n_out, KANN_C_MSE)
        if node == NULL:
            raise KannModelError("Failed to create MSE cost")
        return KadNode.wrap(node)

    def build(self, KadNode cost):
        """
        Build the neural network from the cost node.

        Args:
            cost: The cost/loss node

        Returns:
            NeuralNetwork instance
        """
        cdef kann_t* ann = kann_new(cost._node, 0)
        if ann == NULL:
            raise KannModelError("Failed to build neural network")
        return NeuralNetwork._wrap(ann)


# ============================================================================
# Data Loading (kann_extra)
# ============================================================================

cdef class DataSet:
    """
    Wrapper for KANN tabular data loading (kann_data_t).

    Reads TSV (tab-separated values) files with optional row/column names
    and group boundaries. This is useful for loading training data for
    neural networks.

    File Format:
        - First line starting with '#' contains column names (tab-separated)
        - Each subsequent line: row_name<TAB>value1<TAB>value2<TAB>...
        - Empty lines create group boundaries

    Example file (data.tsv):
        #	feature1	feature2	label
        sample1	0.5	0.3	1.0
        sample2	0.2	0.8	0.0

        sample3	0.9	0.1	1.0

    Usage:
        data = DataSet.load("data.tsv")
        print(f"Loaded {data.n_rows} samples with {data.n_cols} features")

        # Access data
        for i in range(data.n_rows):
            row = data.get_row(i)
            name = data.get_row_name(i)

        # Get all data as 2D list
        x, y = data.split_xy(label_cols=1)
    """
    cdef kann_data_t* _data
    cdef bint _owned

    def __cinit__(self):
        self._data = NULL
        self._owned = False

    def __dealloc__(self):
        if self._data != NULL and self._owned:
            kann_data_free(self._data)
            self._data = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly release resources."""
        if self._data != NULL and self._owned:
            kann_data_free(self._data)
            self._data = NULL
            self._owned = False

    @staticmethod
    def load(str filename):
        """
        Load tabular data from a TSV file.

        Args:
            filename: Path to TSV file

        Returns:
            DataSet instance

        Raises:
            KannError: If file cannot be loaded
        """
        import os
        if not os.path.exists(filename):
            raise KannError(f"Data file not found: {filename}")

        cdef bytes fn_bytes = filename.encode('utf-8')
        cdef kann_data_t* data = kann_data_read(fn_bytes)
        if data == NULL:
            raise KannError(f"Failed to load data from {filename}")

        cdef DataSet ds = DataSet.__new__(DataSet)
        ds._data = data
        ds._owned = True
        return ds

    @property
    def n_rows(self):
        """Number of rows (samples)."""
        if self._data == NULL:
            return 0
        return self._data.n_row

    @property
    def n_cols(self):
        """Number of columns (features)."""
        if self._data == NULL:
            return 0
        return self._data.n_col

    @property
    def n_groups(self):
        """Number of groups."""
        if self._data == NULL:
            return 0
        return self._data.n_grp

    @property
    def shape(self):
        """Shape as (n_rows, n_cols) tuple."""
        return (self.n_rows, self.n_cols)

    def get_row(self, int index):
        """
        Get a single row of data.

        Args:
            index: Row index (0-based)

        Returns:
            array.array('f', ...) containing the row data
        """
        if self._data == NULL:
            raise KannError("No data loaded")
        if index < 0 or index >= self._data.n_row:
            raise IndexError(f"Row index {index} out of range [0, {self._data.n_row})")

        cdef int j
        cdef int n_col = self._data.n_col
        row = array.array('f', [0.0] * n_col)
        cdef float[:] row_view = row
        for j in range(n_col):
            row_view[j] = self._data.x[index][j]
        return row

    def get_row_name(self, int index):
        """
        Get the name of a row.

        Args:
            index: Row index (0-based)

        Returns:
            Row name as string, or None if no names
        """
        if self._data == NULL:
            raise KannError("No data loaded")
        if index < 0 or index >= self._data.n_row:
            raise IndexError(f"Row index {index} out of range")
        if self._data.rname == NULL:
            return None
        if self._data.rname[index] == NULL:
            return None
        return self._data.rname[index].decode('utf-8')

    def get_col_name(self, int index):
        """
        Get the name of a column.

        Args:
            index: Column index (0-based)

        Returns:
            Column name as string, or None if no names
        """
        if self._data == NULL:
            raise KannError("No data loaded")
        if index < 0 or index >= self._data.n_col:
            raise IndexError(f"Column index {index} out of range")
        if self._data.cname == NULL:
            return None
        if self._data.cname[index] == NULL:
            return None
        return self._data.cname[index].decode('utf-8')

    @property
    def row_names(self):
        """List of all row names."""
        if self._data == NULL or self._data.rname == NULL:
            return None
        return [self.get_row_name(i) for i in range(self._data.n_row)]

    @property
    def col_names(self):
        """List of all column names."""
        if self._data == NULL or self._data.cname == NULL:
            return None
        return [self.get_col_name(i) for i in range(self._data.n_col)]

    def get_group_sizes(self):
        """
        Get the size of each group.

        Returns:
            List of group sizes
        """
        if self._data == NULL:
            return []
        if self._data.grp == NULL:
            return [self._data.n_row]
        return [self._data.grp[i] for i in range(self._data.n_grp)]

    def to_2d_array(self):
        """
        Convert all data to a flat array suitable for training.

        Returns:
            Array2D containing all data
        """
        if self._data == NULL:
            raise KannError("No data loaded")

        cdef int i, j
        cdef int n_row = self._data.n_row
        cdef int n_col = self._data.n_col

        flat = array.array('f', [0.0] * (n_row * n_col))
        cdef float[:] flat_view = flat

        for i in range(n_row):
            for j in range(n_col):
                flat_view[i * n_col + j] = self._data.x[i][j]

        return Array2D(n_row, n_col, flat)

    def split_xy(self, int label_cols=1):
        """
        Split data into features (X) and labels (Y).

        Args:
            label_cols: Number of columns at the end to use as labels

        Returns:
            Tuple of (x_array, y_array) as Array2D objects
        """
        if self._data == NULL:
            raise KannError("No data loaded")

        cdef int n_row = self._data.n_row
        cdef int n_col = self._data.n_col
        cdef int x_cols = n_col - label_cols

        if x_cols <= 0:
            raise ValueError(f"label_cols ({label_cols}) must be less than n_cols ({n_col})")

        cdef int i, j

        # Create X array
        x_flat = array.array('f', [0.0] * (n_row * x_cols))
        cdef float[:] x_view = x_flat
        for i in range(n_row):
            for j in range(x_cols):
                x_view[i * x_cols + j] = self._data.x[i][j]

        # Create Y array
        y_flat = array.array('f', [0.0] * (n_row * label_cols))
        cdef float[:] y_view = y_flat
        for i in range(n_row):
            for j in range(label_cols):
                y_view[i * label_cols + j] = self._data.x[i][x_cols + j]

        return Array2D(n_row, x_cols, x_flat), Array2D(n_row, label_cols, y_flat)

    def __len__(self):
        return self.n_rows

    def __getitem__(self, int index):
        return self.get_row(index)

    def __iter__(self):
        """Iterate over rows."""
        for i in range(self.n_rows):
            yield self.get_row(i)

    def __repr__(self):
        if self._data == NULL:
            return "DataSet(empty)"
        return f"DataSet(rows={self.n_rows}, cols={self.n_cols}, groups={self.n_groups})"
