"""Tests for the KANN neural network Cython wrapper.

This module tests the coremusic.kann wrapper for the KANN library,
including network creation, training, and inference.

The wrapper uses memoryviews and the buffer protocol, so it works with
array.array, numpy.ndarray, or any buffer-compatible type.
"""

import os
import tempfile
import array
import random
import pytest

# Try to import numpy for optional tests
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestKannImport:
    """Test basic import and constants."""

    def test_import_kann(self):
        """Test that kann module can be imported."""
        from coremusic import kann
        assert kann is not None

    def test_constants_available(self):
        """Test that constants are exported."""
        from coremusic import kann

        # Cost types
        assert kann.COST_BINARY_CROSS_ENTROPY == 1
        assert kann.COST_MULTI_CROSS_ENTROPY == 2
        assert kann.COST_BINARY_CROSS_ENTROPY_NEG == 3
        assert kann.COST_MSE == 4

        # Flags
        assert kann.KANN_FLAG_IN == 0x1
        assert kann.KANN_FLAG_OUT == 0x2
        assert kann.KANN_FLAG_TRUTH == 0x4
        assert kann.KANN_FLAG_COST == 0x8

        # RNN flags
        assert kann.RNN_VAR_H0 == 0x1
        assert kann.RNN_NORM == 0x2

    def test_exception_classes(self):
        """Test that exception classes are defined."""
        from coremusic.kann import KannError, KannModelError, KannTrainingError

        assert issubclass(KannModelError, KannError)
        assert issubclass(KannTrainingError, KannError)


class TestNeuralNetworkCreation:
    """Test neural network creation."""

    def test_create_mlp(self):
        """Test creating a simple MLP."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(
            input_size=10,
            hidden_sizes=[20, 10],
            output_size=5
        )

        assert nn is not None
        assert nn.n_nodes > 0
        assert nn.input_dim == 10
        assert nn.output_dim == 5
        assert nn.n_var > 0

    def test_create_mlp_with_dropout(self):
        """Test creating MLP with dropout."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(
            input_size=10,
            hidden_sizes=[20],
            output_size=5,
            dropout=0.5
        )

        assert nn is not None
        assert nn.n_nodes > 0

    def test_create_mlp_mse(self):
        """Test creating MLP with MSE loss."""
        from coremusic.kann import NeuralNetwork, COST_MSE

        nn = NeuralNetwork.mlp(
            input_size=10,
            hidden_sizes=[20],
            output_size=5,
            cost_type=COST_MSE
        )

        assert nn is not None

    def test_create_lstm(self):
        """Test creating an LSTM network."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.lstm(
            input_size=10,
            hidden_size=20,
            output_size=10
        )

        assert nn is not None
        assert nn.n_nodes > 0

    def test_create_gru(self):
        """Test creating a GRU network."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.gru(
            input_size=10,
            hidden_size=20,
            output_size=10
        )

        assert nn is not None
        assert nn.n_nodes > 0

    def test_create_rnn(self):
        """Test creating a simple RNN."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.rnn(
            input_size=10,
            hidden_size=20,
            output_size=10
        )

        assert nn is not None
        assert nn.n_nodes > 0

    def test_context_manager(self):
        """Test using network as context manager."""
        from coremusic.kann import NeuralNetwork

        with NeuralNetwork.mlp(10, [20], 5) as nn:
            assert nn.n_nodes > 0

    def test_clone(self):
        """Test cloning a network."""
        from coremusic.kann import NeuralNetwork

        nn1 = NeuralNetwork.mlp(10, [20], 5)
        nn2 = nn1.clone(batch_size=32)

        assert nn2 is not None
        assert nn2.n_nodes == nn1.n_nodes


class TestNeuralNetworkInferenceArrayArray:
    """Test neural network inference with array.array."""

    def test_apply_single(self):
        """Test applying network to single input using array.array."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(10, [20], 5)

        # Use array.array instead of numpy
        x = array.array('f', [random.gauss(0, 1) for _ in range(10)])
        y = nn.apply(x)

        assert y is not None
        assert len(y) == 5
        assert isinstance(y, array.array)

    def test_apply_multiple(self):
        """Test applying network to multiple inputs."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(10, [20], 5)

        for _ in range(10):
            x = array.array('f', [random.gauss(0, 1) for _ in range(10)])
            y = nn.apply(x)
            assert len(y) == 5

    def test_softmax_output(self):
        """Test that softmax output sums to ~1."""
        from coremusic.kann import NeuralNetwork, COST_MULTI_CROSS_ENTROPY

        nn = NeuralNetwork.mlp(
            10, [20], 5,
            cost_type=COST_MULTI_CROSS_ENTROPY
        )

        x = array.array('f', [random.gauss(0, 1) for _ in range(10)])
        y = nn.apply(x)

        # Softmax should sum to approximately 1
        total = sum(y)
        assert abs(total - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
class TestNeuralNetworkInferenceNumpy:
    """Test neural network inference with numpy arrays."""

    def test_apply_single_numpy(self):
        """Test applying network to single input using numpy."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(10, [20], 5)

        x = np.random.randn(10).astype(np.float32)
        y = nn.apply(x)

        assert y is not None
        assert len(y) == 5

    def test_apply_numpy_float32(self):
        """Test that numpy float32 arrays work correctly."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(10, [20], 5)

        x = np.zeros(10, dtype=np.float32)
        x[0] = 1.0
        y = nn.apply(x)

        assert len(y) == 5


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available for 2D memoryview casting")
class TestNeuralNetworkTrainingArrayArray:
    """Test neural network training with array.array (requires numpy for 2D views)."""

    def test_train_xor(self):
        """Test training on XOR problem using numpy (array.array 2D views are tricky)."""
        from coremusic.kann import NeuralNetwork, COST_MSE

        # XOR dataset - use numpy for reliable 2D array handling
        x_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.float32)

        y_train = np.array([
            [0],
            [1],
            [1],
            [0],
        ], dtype=np.float32)

        nn = NeuralNetwork.mlp(2, [4], 1, cost_type=COST_MSE)

        epochs = nn.train(
            x_train, y_train,
            learning_rate=0.1,
            mini_batch_size=4,
            max_epochs=1000,
            validation_fraction=0.0
        )

        assert epochs > 0

    def test_compute_cost(self):
        """Test computing cost on a dataset."""
        from coremusic.kann import NeuralNetwork, COST_MSE

        nn = NeuralNetwork.mlp(10, [20], 5, cost_type=COST_MSE)

        # Create random data with numpy
        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 5).astype(np.float32)

        cost = nn.cost(x, y)
        assert cost > 0


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
class TestNeuralNetworkTrainingNumpy:
    """Test neural network training with numpy arrays."""

    def test_train_xor_numpy(self):
        """Test training on XOR problem with numpy."""
        from coremusic.kann import NeuralNetwork, COST_MSE

        # XOR dataset
        x_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.float32)

        y_train = np.array([
            [0],
            [1],
            [1],
            [0],
        ], dtype=np.float32)

        nn = NeuralNetwork.mlp(2, [4], 1, cost_type=COST_MSE)

        epochs = nn.train(
            x_train, y_train,
            learning_rate=0.1,
            mini_batch_size=4,
            max_epochs=1000,
            validation_fraction=0.0
        )

        assert epochs > 0

    def test_train_classification_numpy(self):
        """Test training a classification network with numpy."""
        from coremusic.kann import NeuralNetwork, COST_MULTI_CROSS_ENTROPY

        # Simple 3-class classification
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        n_classes = 3

        x_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.zeros((n_samples, n_classes), dtype=np.float32)
        labels = np.random.randint(0, n_classes, n_samples)
        for i, label in enumerate(labels):
            y_train[i, label] = 1.0

        nn = NeuralNetwork.mlp(
            n_features, [20], n_classes,
            cost_type=COST_MULTI_CROSS_ENTROPY
        )

        epochs = nn.train(
            x_train, y_train,
            learning_rate=0.01,
            mini_batch_size=32,
            max_epochs=50,
            validation_fraction=0.1
        )

        assert epochs > 0


class TestModelIO:
    """Test model save/load."""

    def test_save_load(self):
        """Test saving and loading a model."""
        from coremusic.kann import NeuralNetwork

        nn1 = NeuralNetwork.mlp(10, [20], 5)

        # Apply to get initial output
        x = array.array('f', [1.0] * 10)
        y1 = nn1.apply(x)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.kan', delete=False) as f:
            model_path = f.name

        try:
            nn1.save(model_path)
            nn2 = NeuralNetwork.load(model_path)

            # Outputs should be identical
            y2 = nn2.apply(x)

            for i in range(len(y1)):
                assert abs(y1[i] - y2[i]) < 1e-6

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_load_nonexistent(self):
        """Test loading a nonexistent model."""
        from coremusic.kann import NeuralNetwork, KannModelError

        with pytest.raises(KannModelError):
            NeuralNetwork.load("/nonexistent/path/model.kan")


class TestGraphBuilder:
    """Test the low-level graph builder."""

    def test_build_simple_network(self):
        """Test building a simple network with GraphBuilder."""
        from coremusic.kann import GraphBuilder

        builder = GraphBuilder()
        x = builder.input(10)
        h = builder.dense(x, 20)
        h = builder.relu(h)
        h = builder.dense(h, 10)
        cost = builder.softmax_cross_entropy(h, 5)

        nn = builder.build(cost)
        assert nn is not None
        assert nn.n_nodes > 0

    def test_build_with_dropout(self):
        """Test building network with dropout."""
        from coremusic.kann import GraphBuilder

        builder = GraphBuilder()
        x = builder.input(10)
        h = builder.dense(x, 20)
        h = builder.relu(h)
        h = builder.dropout(h, 0.5)
        h = builder.dense(h, 10)
        cost = builder.softmax_cross_entropy(h, 5)

        nn = builder.build(cost)
        assert nn is not None

    def test_build_with_layernorm(self):
        """Test building network with layer normalization."""
        from coremusic.kann import GraphBuilder

        builder = GraphBuilder()
        x = builder.input(10)
        h = builder.dense(x, 20)
        h = builder.layernorm(h)
        h = builder.relu(h)
        cost = builder.softmax_cross_entropy(h, 5)

        nn = builder.build(cost)
        assert nn is not None

    def test_activations(self):
        """Test different activation functions."""
        from coremusic.kann import GraphBuilder

        builder = GraphBuilder()
        x = builder.input(10)

        # Test all activations
        h = builder.dense(x, 20)
        _ = builder.relu(h)
        _ = builder.sigmoid(h)
        _ = builder.tanh(h)
        _ = builder.softmax(h)


class TestUtilities:
    """Test utility functions."""

    def test_set_seed(self):
        """Test setting random seed."""
        from coremusic.kann import set_seed

        set_seed(42)
        # No error means success

    def test_set_verbose(self):
        """Test setting verbosity."""
        from coremusic.kann import set_verbose

        set_verbose(0)
        set_verbose(1)
        # No error means success

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        from coremusic.kann import one_hot_encode

        values = array.array('i', [0, 1, 2, 0, 1])
        encoded = one_hot_encode(values, 3)

        assert len(encoded) == 5
        assert len(encoded[0]) == 3

        # Check encoding is correct
        assert encoded[0][0] == 1.0 and encoded[0][1] == 0.0 and encoded[0][2] == 0.0
        assert encoded[1][0] == 0.0 and encoded[1][1] == 1.0 and encoded[1][2] == 0.0
        assert encoded[2][0] == 0.0 and encoded[2][1] == 0.0 and encoded[2][2] == 1.0

    def test_one_hot_encode_2d(self):
        """Test one-hot encoding to flat 2D array."""
        from coremusic.kann import one_hot_encode_2d

        values = array.array('i', [0, 1, 2])
        encoded = one_hot_encode_2d(values, 3)

        assert len(encoded) == 9  # 3 values * 3 classes
        # First row: [1, 0, 0]
        assert encoded[0] == 1.0
        assert encoded[1] == 0.0
        assert encoded[2] == 0.0
        # Second row: [0, 1, 0]
        assert encoded[3] == 0.0
        assert encoded[4] == 1.0
        assert encoded[5] == 0.0

    def test_softmax_sample(self):
        """Test softmax sampling."""
        from coremusic.kann import softmax_sample

        probs = array.array('f', [0.1, 0.2, 0.7])

        # Sample many times and check distribution
        random.seed(42)
        samples = [softmax_sample(probs) for _ in range(1000)]

        # Most samples should be index 2 (highest probability)
        counts = [samples.count(i) for i in range(3)]
        assert counts[2] > counts[0]
        assert counts[2] > counts[1]

    def test_softmax_sample_temperature(self):
        """Test softmax sampling with temperature."""
        from coremusic.kann import softmax_sample

        probs = array.array('f', [0.1, 0.2, 0.7])

        # High temperature should make distribution more uniform
        random.seed(42)
        samples_high_temp = [softmax_sample(probs, temperature=10.0) for _ in range(1000)]

        # Low temperature should concentrate on max
        random.seed(42)
        samples_low_temp = [softmax_sample(probs, temperature=0.1) for _ in range(1000)]

        # High temp should have more variety
        high_temp_counts = [samples_high_temp.count(i) for i in range(3)]
        low_temp_counts = [samples_low_temp.count(i) for i in range(3)]

        # Low temperature should have most samples at index 2
        assert low_temp_counts[2] > high_temp_counts[2]

    def test_prepare_sequence_data(self):
        """Test preparing sequence data for RNN training."""
        from coremusic.kann import prepare_sequence_data

        sequences = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ]
        seq_length = 3
        vocab_size = 5

        x_train, y_train = prepare_sequence_data(sequences, seq_length, vocab_size)

        # Each sequence of length 5 gives 1 training example with seq_length=3
        assert len(x_train) == len(y_train)
        assert len(x_train[0]) == seq_length * vocab_size
        assert len(y_train[0]) == vocab_size


class TestArray2D:
    """Test the Array2D helper class."""

    def test_create_empty(self):
        """Test creating empty Array2D."""
        from coremusic.kann import Array2D

        arr = Array2D(10, 5)
        assert arr.rows == 10
        assert arr.cols == 5
        assert len(arr.data) == 50

    def test_create_with_data(self):
        """Test creating Array2D with initial data."""
        from coremusic.kann import Array2D

        arr = Array2D(2, 3, [1, 2, 3, 4, 5, 6])
        assert arr[0, 0] == 1
        assert arr[0, 2] == 3
        assert arr[1, 0] == 4
        assert arr[1, 2] == 6

    def test_setitem(self):
        """Test setting values in Array2D."""
        from coremusic.kann import Array2D

        arr = Array2D(2, 2)
        arr[0, 0] = 1.0
        arr[0, 1] = 2.0
        arr[1, 0] = 3.0
        arr[1, 1] = 4.0

        assert arr[0, 0] == 1.0
        assert arr[0, 1] == 2.0
        assert arr[1, 0] == 3.0
        assert arr[1, 1] == 4.0

    def test_from_list(self):
        """Test creating Array2D from list of lists."""
        from coremusic.kann import Array2D

        arr = Array2D.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        assert arr.rows == 2
        assert arr.cols == 3
        assert arr[0, 0] == 1
        assert arr[1, 2] == 6


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
class TestMIDIGeneration:
    """Test MIDI generation workflow."""

    def test_midi_note_prediction(self):
        """Test training a network for MIDI note prediction."""
        from coremusic.kann import NeuralNetwork, COST_MULTI_CROSS_ENTROPY

        # Simulate MIDI note sequences (128 possible notes)
        vocab_size = 128
        np.random.seed(42)

        # Create simple sequence data
        n_samples = 100

        x_train = np.random.randn(n_samples, vocab_size).astype(np.float32)

        y_train = np.zeros((n_samples, vocab_size), dtype=np.float32)
        targets = np.random.randint(0, vocab_size, n_samples)
        for i, t in enumerate(targets):
            y_train[i, t] = 1.0

        # Create and train network
        nn = NeuralNetwork.mlp(
            input_size=vocab_size,
            hidden_sizes=[64],
            output_size=vocab_size,
            cost_type=COST_MULTI_CROSS_ENTROPY
        )

        epochs = nn.train(
            x_train, y_train,
            learning_rate=0.01,
            mini_batch_size=32,
            max_epochs=10,
            validation_fraction=0.1
        )

        assert epochs > 0

        # Test inference with array.array (1D works fine)
        test_input = array.array('f', [0.0] * vocab_size)
        test_input[60] = 1.0  # Middle C
        output = nn.apply(test_input)

        assert len(output) == vocab_size
        total = sum(output)
        assert abs(total - 1.0) < 1e-5  # Should be probability distribution


class TestRNNOperations:
    """Test RNN-specific operations."""

    def test_unroll_lstm(self):
        """Test unrolling an LSTM network."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.lstm(10, 20, 10)
        unrolled = nn.unroll(5)

        assert unrolled is not None
        assert unrolled.n_nodes > nn.n_nodes

    def test_rnn_start_end(self):
        """Test RNN start/end for continuous feeding."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.lstm(10, 20, 10)

        nn.rnn_start()
        nn.rnn_end()
        # No error means success

    def test_switch_mode(self):
        """Test switching between training and inference mode."""
        from coremusic.kann import NeuralNetwork

        nn = NeuralNetwork.mlp(10, [20], 5, dropout=0.5)

        # Switch to training mode
        nn.switch_mode(is_training=True)

        # Switch to inference mode
        nn.switch_mode(is_training=False)

        # Inference should work
        x = array.array('f', [random.gauss(0, 1) for _ in range(10)])
        y = nn.apply(x)
        assert y is not None


class TestDataSet:
    """Test the DataSet class for loading tabular data."""

    def test_load_tsv_file(self):
        """Test loading a TSV file."""
        from coremusic.kann import DataSet, KannError

        # Create a temporary TSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("#\tfeature1\tfeature2\tlabel\n")
            f.write("sample1\t0.5\t0.3\t1.0\n")
            f.write("sample2\t0.2\t0.8\t0.0\n")
            f.write("sample3\t0.9\t0.1\t1.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)

            assert data.n_rows == 3
            assert data.n_cols == 3
            assert data.shape == (3, 3)

            # Check row access
            row0 = data.get_row(0)
            assert len(row0) == 3
            assert abs(row0[0] - 0.5) < 0.01
            assert abs(row0[1] - 0.3) < 0.01
            assert abs(row0[2] - 1.0) < 0.01

            # Check names
            assert data.get_row_name(0) == "sample1"
            assert data.get_row_name(1) == "sample2"

            col_names = data.col_names
            assert col_names == ["feature1", "feature2", "label"]

        finally:
            os.unlink(tsv_path)

    def test_dataset_context_manager(self):
        """Test DataSet as context manager."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\n")
            f.write("row2\t3.0\t4.0\n")
            tsv_path = f.name

        try:
            with DataSet.load(tsv_path) as data:
                assert data.n_rows == 2
                assert data.n_cols == 2
        finally:
            os.unlink(tsv_path)

    def test_dataset_iteration(self):
        """Test iterating over DataSet rows."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\n")
            f.write("row2\t3.0\t4.0\n")
            f.write("row3\t5.0\t6.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)

            rows = list(data)
            assert len(rows) == 3

            # Check values
            assert abs(rows[0][0] - 1.0) < 0.01
            assert abs(rows[1][0] - 3.0) < 0.01
            assert abs(rows[2][0] - 5.0) < 0.01

        finally:
            os.unlink(tsv_path)

    def test_dataset_to_2d_array(self):
        """Test converting DataSet to Array2D."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\n")
            f.write("row2\t3.0\t4.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)
            arr = data.to_2d_array()

            assert arr.rows == 2
            assert arr.cols == 2
            assert arr[0, 0] == 1.0
            assert arr[0, 1] == 2.0
            assert arr[1, 0] == 3.0
            assert arr[1, 1] == 4.0

        finally:
            os.unlink(tsv_path)

    def test_dataset_split_xy(self):
        """Test splitting DataSet into X and Y."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\t0.0\n")
            f.write("row2\t3.0\t4.0\t1.0\n")
            f.write("row3\t5.0\t6.0\t0.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)
            x, y = data.split_xy(label_cols=1)

            assert x.rows == 3
            assert x.cols == 2  # 3 cols - 1 label col
            assert y.rows == 3
            assert y.cols == 1

            # Check X values
            assert x[0, 0] == 1.0
            assert x[0, 1] == 2.0

            # Check Y values
            assert y[0, 0] == 0.0
            assert y[1, 0] == 1.0

        finally:
            os.unlink(tsv_path)

    def test_dataset_load_nonexistent(self):
        """Test loading nonexistent file."""
        from coremusic.kann import DataSet, KannError

        with pytest.raises(KannError):
            DataSet.load("/nonexistent/path/data.tsv")

    def test_dataset_groups(self):
        """Test DataSet with group boundaries."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\n")
            f.write("row2\t3.0\t4.0\n")
            f.write("\n")  # Empty line creates group boundary
            f.write("row3\t5.0\t6.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)

            assert data.n_rows == 3
            assert data.n_groups >= 1  # At least one group

            group_sizes = data.get_group_sizes()
            assert len(group_sizes) > 0

        finally:
            os.unlink(tsv_path)

    def test_dataset_repr(self):
        """Test DataSet string representation."""
        from coremusic.kann import DataSet

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("row1\t1.0\t2.0\n")
            tsv_path = f.name

        try:
            data = DataSet.load(tsv_path)
            repr_str = repr(data)

            assert "DataSet" in repr_str
            assert "rows=1" in repr_str
            assert "cols=2" in repr_str

        finally:
            os.unlink(tsv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
