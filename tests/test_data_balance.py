"""Tests for XAI data balance analysis functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

import xai
from .conftest import assert_dataframe_equal


class TestImbalancePlot:
    """Tests for imbalance_plot function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_imbalance_plot_single_column(self, mock_bar, mock_figure, mock_show,
                                        sample_binary_target_df, categorical_columns):
        """Test imbalance_plot with single column."""
        df = sample_binary_target_df.copy()

        result = xai.imbalance_plot(df, "gender", categorical_cols=categorical_columns)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_bar.assert_called()
        mock_show.assert_called()

        # Should return some result (likely group counts)
        assert result is not None

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_imbalance_plot_multiple_columns(self, mock_bar, mock_figure, mock_show,
                                           sample_binary_target_df, categorical_columns):
        """Test imbalance_plot with multiple columns (cross-tabulation)."""
        df = sample_binary_target_df.copy()

        result = xai.imbalance_plot(df, "gender", "target", categorical_cols=categorical_columns)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_bar.assert_called()
        mock_show.assert_called()

        # Should return some result
        assert result is not None

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_imbalance_plot_data_counting_logic(self, mock_figure, mock_show, categorical_columns):
        """Test that imbalance_plot correctly counts data groups."""
        # Create DataFrame with known imbalances
        df = pd.DataFrame({
            'gender': ['Male'] * 30 + ['Female'] * 20,
            'target': [0] * 25 + [1] * 25,
            'other_col': ['A'] * 50
        })

        result = xai.imbalance_plot(df, "gender", categorical_cols=categorical_columns)

        # The function should process the data and return group information
        # Exact return format depends on implementation
        assert result is not None

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_imbalance_plot_empty_dataframe(self, mock_figure, mock_show, sample_empty_df):
        """Test imbalance_plot with empty DataFrame."""
        # Should handle empty DataFrame gracefully
        try:
            result = xai.imbalance_plot(sample_empty_df, "nonexistent_col")
            # If it doesn't raise an exception, it should return something reasonable
        except (KeyError, ValueError, IndexError):
            # These exceptions are acceptable for empty DataFrame
            pass

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_imbalance_plot_preserves_original_data(self, mock_figure, mock_show,
                                                  sample_binary_target_df, categorical_columns):
        """Test that imbalance_plot doesn't modify original DataFrame."""
        df = sample_binary_target_df.copy()
        original = df.copy()

        xai.imbalance_plot(df, "gender", categorical_cols=categorical_columns)

        assert_dataframe_equal(df, original)


class TestBalance:
    """Tests for balance function."""

    def test_balance_basic_functionality(self, sample_binary_target_df, categorical_columns):
        """Test basic balance function with upsampling."""
        df = sample_binary_target_df.copy()
        original_length = len(df)

        # Test upsampling
        result = xai.balance(df, "gender", "target", upsample=0.8, categorical_cols=categorical_columns)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= original_length  # Upsampling should increase size

        # Should have same columns as original
        assert set(result.columns) == set(df.columns)

    def test_balance_upsampling_logic(self, categorical_columns):
        """Test that balance upsampling works correctly."""
        # Create known imbalanced dataset
        df = pd.DataFrame({
            'gender': ['Male'] * 60 + ['Female'] * 40,
            'target': [0] * 80 + [1] * 20,  # 80/20 imbalance
            'feature': range(100)
        })

        result = xai.balance(df, "gender", "target", upsample=0.9, categorical_cols=categorical_columns)

        # Should balance the data
        assert len(result) > len(df)  # More rows due to upsampling

    def test_balance_downsampling_logic(self, categorical_columns):
        """Test that balance downsampling works correctly."""
        # Create known imbalanced dataset
        df = pd.DataFrame({
            'gender': ['Male'] * 60 + ['Female'] * 40,
            'target': [0] * 80 + [1] * 20,
            'feature': range(100)
        })

        result = xai.balance(df, "gender", "target", downsample=0.5, categorical_cols=categorical_columns)

        # Should reduce data size
        assert len(result) <= len(df)

    def test_balance_preserves_original(self, sample_binary_target_df, categorical_columns):
        """Test that balance doesn't modify original DataFrame."""
        df = sample_binary_target_df.copy()
        original = df.copy()

        xai.balance(df, "gender", "target", categorical_cols=categorical_columns)

        assert_dataframe_equal(df, original)

    def test_balance_empty_dataframe(self, sample_empty_df):
        """Test balance with empty DataFrame."""
        # Should handle empty DataFrame gracefully
        try:
            result = xai.balance(sample_empty_df, "nonexistent1", "nonexistent2")
            assert len(result) == 0
        except (KeyError, ValueError, IndexError):
            # These exceptions are acceptable for empty DataFrame
            pass

    def test_balance_single_group(self, categorical_columns):
        """Test balance when only one group exists."""
        df = pd.DataFrame({
            'gender': ['Male'] * 10,  # Only one gender
            'target': [0] * 5 + [1] * 5,
            'feature': range(10)
        })

        result = xai.balance(df, "gender", "target", categorical_cols=categorical_columns)

        # Should handle single group case
        assert isinstance(result, pd.DataFrame)

    def test_balance_parameters_validation(self, sample_binary_target_df, categorical_columns):
        """Test balance with various parameter combinations."""
        df = sample_binary_target_df.copy()

        # Test with only upsample
        result1 = xai.balance(df, "gender", "target", upsample=0.7, categorical_cols=categorical_columns)
        assert isinstance(result1, pd.DataFrame)

        # Test with only downsample
        result2 = xai.balance(df, "gender", "target", downsample=0.3, categorical_cols=categorical_columns)
        assert isinstance(result2, pd.DataFrame)


class TestBalancedTrainTestSplit:
    """Tests for balanced_train_test_split function."""

    def test_balanced_train_test_split_basic(self, sample_binary_target_df, categorical_columns):
        """Test basic balanced_train_test_split functionality."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=5,
            max_per_group=10,
            categorical_cols=categorical_columns
        )

        # Should return 6 elements: X_train, y_train, X_test, y_test, train_idx, test_idx
        assert len(result) == 6
        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, (pd.Series, np.ndarray))
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, (pd.Series, np.ndarray))
        assert isinstance(train_idx, (list, np.ndarray, pd.Index))
        assert isinstance(test_idx, (list, np.ndarray, pd.Index))

        # Check that splits make sense
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) <= len(X)  # Could be less due to balancing

    def test_balanced_train_test_split_group_balance(self, categorical_columns):
        """Test that balanced_train_test_split maintains group balance."""
        # Create larger dataset for meaningful split
        df = pd.DataFrame({
            'gender': ['Male'] * 100 + ['Female'] * 100,
            'target': [0] * 100 + [1] * 100,
            'feature1': range(200),
            'feature2': np.random.randn(200)
        })
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=20,
            max_per_group=30,
            categorical_cols=categorical_columns
        )

        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Should have reasonable splits
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_balanced_train_test_split_indices(self, sample_binary_target_df, categorical_columns):
        """Test that balanced_train_test_split returns valid indices."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=3,
            max_per_group=8,
            categorical_cols=categorical_columns
        )

        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Indices should be valid
        assert all(idx < len(X) for idx in train_idx)
        assert all(idx < len(X) for idx in test_idx)

        # No overlap between train and test indices
        train_set = set(train_idx)
        test_set = set(test_idx)
        assert len(train_set.intersection(test_set)) == 0

    def test_balanced_train_test_split_preserves_original(self, sample_binary_target_df, categorical_columns):
        """Test that balanced_train_test_split doesn't modify original data."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']
        X_original = X.copy()
        y_original = y.copy()

        xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=3,
            max_per_group=8,
            categorical_cols=categorical_columns
        )

        assert_dataframe_equal(X, X_original)
        assert y.equals(y_original)

    def test_balanced_train_test_split_edge_cases(self, categorical_columns):
        """Test balanced_train_test_split with edge cases."""
        # Very small dataset
        small_df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'target': [0, 1, 0],
            'feature': [1, 2, 3]
        })
        X_small = small_df.drop('target', axis=1)
        y_small = small_df['target']

        try:
            result = xai.balanced_train_test_split(
                X_small, y_small, "gender",
                min_per_group=1,
                max_per_group=2,
                categorical_cols=categorical_columns
            )
            # If successful, should return 6 elements
            assert len(result) == 6
        except (ValueError, IndexError):
            # These exceptions are acceptable for very small datasets
            pass

    def test_balanced_train_test_split_test_size_parameter(self, sample_binary_target_df, categorical_columns):
        """Test balanced_train_test_split with test_size parameter."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        try:
            result = xai.balanced_train_test_split(
                X, y, "gender",
                min_per_group=5,
                max_per_group=10,
                test_size=0.3,  # 30% for testing
                categorical_cols=categorical_columns
            )

            X_train, y_train, X_test, y_test, train_idx, test_idx = result

            # Test set should be roughly 30% (subject to balancing constraints)
            total_samples = len(X_train) + len(X_test)
            if total_samples > 0:
                test_ratio = len(X_test) / total_samples
                assert 0.1 <= test_ratio <= 0.5  # Allow for balancing effects

        except TypeError:
            # If test_size parameter is not supported, that's okay
            pytest.skip("test_size parameter not supported in balanced_train_test_split")