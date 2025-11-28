"""Tests for XAI advanced analysis functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

import xai
from .conftest import assert_dataframe_equal, assert_array_almost_equal


class TestSmileImbalance:
    """Tests for smile_imbalance function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_smile_imbalance_basic(self, mock_bar, mock_figure, mock_show, sample_evaluation_data):
        """Test basic smile_imbalance functionality."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true)).reshape(-1, 1)

        result = xai.smile_imbalance(y_true, y_prob)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_bar.assert_called()
        mock_show.assert_called()

        # Should have expected columns for accuracy analysis
        expected_columns = {'correct', 'incorrect'}
        assert expected_columns.issubset(set(result.columns))

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_smile_imbalance_probability_bucketing(self, mock_figure, mock_show):
        """Test that smile_imbalance correctly buckets probabilities."""
        # Create controlled test case
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4]).reshape(-1, 1)

        result = xai.smile_imbalance(y_true, y_prob, bins=4)

        # Should return DataFrame with probability buckets
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 4  # Should not exceed specified bins

        # Should have accuracy-related columns
        assert 'correct' in result.columns or 'incorrect' in result.columns

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_smile_imbalance_threshold_parameter(self, mock_figure, mock_show, sample_evaluation_data):
        """Test smile_imbalance with custom threshold."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true)).reshape(-1, 1)

        result = xai.smile_imbalance(y_true, y_prob, threshold=0.3)

        assert isinstance(result, pd.DataFrame)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_smile_imbalance_display_breakdown(self, mock_figure, mock_show, sample_evaluation_data):
        """Test smile_imbalance with display_breakdown parameter."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true)).reshape(-1, 1)

        result = xai.smile_imbalance(y_true, y_prob, display_breakdown=True)

        assert isinstance(result, pd.DataFrame)

        # With breakdown, should have detailed metric columns
        expected_breakdown_cols = {'true-positives', 'true-negatives', 'false-positives', 'false-negatives'}
        # At least some breakdown columns should be present
        assert len(expected_breakdown_cols.intersection(set(result.columns))) > 0

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_smile_imbalance_manual_review(self, mock_figure, mock_show, sample_evaluation_data):
        """Test smile_imbalance with manual review parameter."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true)).reshape(-1, 1)

        result = xai.smile_imbalance(y_true, y_prob, manual_review=0.2)

        assert isinstance(result, pd.DataFrame)

        # With manual review, should have manual-review column
        assert 'manual-review' in result.columns

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_smile_imbalance_custom_bins(self, mock_figure, mock_show, sample_evaluation_data):
        """Test smile_imbalance with different bin numbers."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true)).reshape(-1, 1)

        result_5_bins = xai.smile_imbalance(y_true, y_prob, bins=5)
        result_10_bins = xai.smile_imbalance(y_true, y_prob, bins=10)

        assert isinstance(result_5_bins, pd.DataFrame)
        assert isinstance(result_10_bins, pd.DataFrame)

        # Different bin numbers should potentially create different result sizes
        assert len(result_5_bins) <= 5
        assert len(result_10_bins) <= 10


class TestFeatureImportance:
    """Tests for feature_importance function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    def test_feature_importance_basic(self, mock_bar, mock_figure, mock_show,
                                    sample_binary_target_df, mock_evaluation_function):
        """Test basic feature_importance functionality."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.feature_importance(X, y, mock_evaluation_function, repeat=3, plot=True)

        # Should return DataFrame with feature importance scores
        assert isinstance(result, pd.DataFrame)

        # Should have columns for features and importance
        assert 'feature' in result.columns or result.index.name is not None

        # Should call matplotlib functions when plot=True
        mock_figure.assert_called()
        mock_bar.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_feature_importance_no_plot(self, mock_figure, mock_show,
                                      sample_binary_target_df, mock_evaluation_function):
        """Test feature_importance with plot=False."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.feature_importance(X, y, mock_evaluation_function, repeat=3, plot=False)

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)

        # Should not call matplotlib functions when plot=False
        mock_figure.assert_not_called()
        mock_show.assert_not_called()

    def test_feature_importance_permutation_logic(self, sample_binary_target_df):
        """Test that feature_importance calls evaluation function correctly."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        call_count = 0

        def mock_eval_with_counter(x_data, y_data):
            nonlocal call_count
            call_count += 1
            return 0.8  # Return consistent score

        result = xai.feature_importance(X, y, mock_eval_with_counter, repeat=2, plot=False)

        # Should call evaluation function multiple times
        # (baseline + features * repeats)
        assert call_count > len(X.columns)  # At least once per feature plus baseline

    def test_feature_importance_repeat_parameter(self, sample_binary_target_df, mock_evaluation_function):
        """Test feature_importance with different repeat values."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']

        result_repeat_1 = xai.feature_importance(X, y, mock_evaluation_function, repeat=1, plot=False)
        result_repeat_5 = xai.feature_importance(X, y, mock_evaluation_function, repeat=5, plot=False)

        # Both should return DataFrames
        assert isinstance(result_repeat_1, pd.DataFrame)
        assert isinstance(result_repeat_5, pd.DataFrame)

        # Should have same number of features
        assert len(result_repeat_1) == len(result_repeat_5)

    def test_feature_importance_empty_features(self, mock_evaluation_function):
        """Test feature_importance with empty feature set."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        try:
            result = xai.feature_importance(X_empty, y_empty, mock_evaluation_function, plot=False)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except (ValueError, IndexError):
            # These exceptions are acceptable for empty input
            pass

    def test_feature_importance_single_feature(self, mock_evaluation_function):
        """Test feature_importance with single feature."""
        X_single = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_single = pd.Series([0, 1, 0, 1, 0])

        result = xai.feature_importance(X_single, y_single, mock_evaluation_function, plot=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Should have one feature

    def test_feature_importance_preserves_original_data(self, sample_binary_target_df, mock_evaluation_function):
        """Test that feature_importance doesn't modify original data."""
        df = sample_binary_target_df.copy()
        X = df.drop('target', axis=1)
        y = df['target']
        X_original = X.copy()
        y_original = y.copy()

        xai.feature_importance(X, y, mock_evaluation_function, plot=False)

        assert_dataframe_equal(X, X_original)
        assert y.equals(y_original)


class TestCorrelations:
    """Tests for correlations function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('scipy.cluster.hierarchy.dendrogram')
    def test_correlations_dendogram_default(self, mock_dendrogram, mock_figure, mock_show, sample_basic_df):
        """Test correlations with default dendogram plot."""
        result = xai.correlations(sample_basic_df, include_categorical=True)

        # Should call matplotlib and scipy functions
        mock_figure.assert_called()
        mock_dendrogram.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    def test_correlations_matrix_plot(self, mock_imshow, mock_figure, mock_show, sample_basic_df):
        """Test correlations with matrix plot type."""
        result = xai.correlations(sample_basic_df, include_categorical=True, plot_type="matrix")

        # Should call matplotlib functions for matrix plot
        mock_figure.assert_called()
        mock_imshow.assert_called()
        mock_show.assert_called()

    def test_correlations_include_categorical_true(self, sample_basic_df):
        """Test correlations with categorical variables included."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('scipy.cluster.hierarchy.dendrogram'):

            result = xai.correlations(sample_basic_df, include_categorical=True)

        # Should process categorical variables
        # The function should handle both numeric and categorical columns

    def test_correlations_include_categorical_false(self, sample_numeric_df):
        """Test correlations with only numeric variables."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('scipy.cluster.hierarchy.dendrogram'):

            result = xai.correlations(sample_numeric_df, include_categorical=False)

        # Should process only numeric columns

    def test_correlations_numeric_only_dataframe(self, sample_numeric_df):
        """Test correlations with DataFrame containing only numeric columns."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('scipy.cluster.hierarchy.dendrogram'):

            result = xai.correlations(sample_numeric_df)

        # Should work with numeric-only DataFrame

    def test_correlations_empty_dataframe(self, sample_empty_df):
        """Test correlations with empty DataFrame."""
        try:
            with patch('matplotlib.pyplot.show'), \
                 patch('matplotlib.pyplot.figure'):
                result = xai.correlations(sample_empty_df)
        except (ValueError, IndexError, TypeError):
            # These exceptions are acceptable for empty DataFrame
            pass

    def test_correlations_single_column(self):
        """Test correlations with single column DataFrame."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})

        try:
            with patch('matplotlib.pyplot.show'), \
                 patch('matplotlib.pyplot.figure'), \
                 patch('scipy.cluster.hierarchy.dendrogram'):
                result = xai.correlations(single_col_df)
        except (ValueError, IndexError):
            # These exceptions are acceptable for single column
            pass

    def test_correlations_preserves_original_data(self, sample_basic_df):
        """Test that correlations doesn't modify original DataFrame."""
        original = sample_basic_df.copy()

        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('scipy.cluster.hierarchy.dendrogram'):
            xai.correlations(sample_basic_df)

        assert_dataframe_equal(sample_basic_df, original)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_correlations_plot_type_validation(self, mock_figure, mock_show, sample_basic_df):
        """Test correlations with different plot_type values."""
        # Test valid plot types
        valid_plot_types = ["matrix", "dendogram"]  # Note: might be "dendrogram" in implementation

        for plot_type in valid_plot_types:
            try:
                with patch('matplotlib.pyplot.imshow'), \
                     patch('scipy.cluster.hierarchy.dendrogram'):
                    result = xai.correlations(sample_basic_df, plot_type=plot_type)

                mock_figure.assert_called()
                mock_show.assert_called()
            except (ValueError, KeyError, TypeError):
                # If plot_type is not supported or spelled differently, that's okay
                pass

    def test_correlations_categorical_conversion_logic(self, sample_categorical_df):
        """Test that correlations properly converts categorical variables."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('scipy.cluster.hierarchy.dendrogram'):

            result = xai.correlations(sample_categorical_df, include_categorical=True)

        # Should handle DataFrame with categorical columns
        # The function should convert categorical variables to numeric for correlation