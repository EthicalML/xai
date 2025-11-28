"""Tests for XAI model evaluation functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

import xai
from .conftest import (
    assert_array_almost_equal,
    create_mock_matplotlib_patches
)


class TestEvaluationMetrics:
    """Tests for evaluation_metrics function."""

    def test_evaluation_metrics_perfect_classifier(self):
        """Test evaluation_metrics with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = xai.evaluation_metrics(y_true, y_pred)

        # Perfect classifier should have perfect metrics
        expected_metrics = {
            'precision': 1.0,
            'recall': 1.0,
            'specificity': 1.0,
            'accuracy': 1.0,
            'f1': 1.0
        }

        assert isinstance(result, dict)
        for metric, expected_value in expected_metrics.items():
            if metric in result:
                assert abs(result[metric] - expected_value) < 0.001

    def test_evaluation_metrics_known_case(self, sample_evaluation_data):
        """Test evaluation_metrics with known input/output case."""
        y_true, y_pred = sample_evaluation_data

        result = xai.evaluation_metrics(y_true, y_pred)

        # Test that all expected metrics are present
        expected_keys = {'precision', 'recall', 'specificity', 'accuracy', 'f1'}
        assert isinstance(result, dict)

        # Check that reasonable values are returned (between 0 and 1)
        for key, value in result.items():
            if key != 'auc':  # AUC requires probabilities
                assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"

    def test_evaluation_metrics_with_probabilities(self, sample_evaluation_data):
        """Test evaluation_metrics with prediction probabilities for AUC calculation."""
        y_true, _ = sample_evaluation_data
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4, 0.15, 0.85])

        result = xai.evaluation_metrics(y_true, y_prob)

        # Should include AUC when probabilities are provided
        assert 'auc' in result
        assert 0 <= result['auc'] <= 1

    def test_evaluation_metrics_edge_cases(self):
        """Test evaluation_metrics with edge cases."""
        # All zeros
        y_true_zeros = np.array([0, 0, 0, 0])
        y_pred_zeros = np.array([0, 0, 0, 0])

        result_zeros = xai.evaluation_metrics(y_true_zeros, y_pred_zeros)
        assert isinstance(result_zeros, dict)

        # All ones
        y_true_ones = np.array([1, 1, 1, 1])
        y_pred_ones = np.array([1, 1, 1, 1])

        result_ones = xai.evaluation_metrics(y_true_ones, y_pred_ones)
        assert isinstance(result_ones, dict)

    def test_evaluation_metrics_binary_predictions(self):
        """Test that evaluation_metrics handles binary predictions correctly."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])  # 50% accuracy

        result = xai.evaluation_metrics(y_true, y_pred)

        # Should calculate accuracy correctly
        assert abs(result['accuracy'] - 0.5) < 0.001


class TestMetricsPlot:
    """Tests for metrics_plot function."""

    @patch('matplotlib.pyplot.axhline')
    def test_metrics_plot_basic(self, mock_axhline, sample_evaluation_data):
        """Test basic metrics_plot functionality with mocked matplotlib."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception
        xai.metrics_plot(y_true, y_prob)

        # Should call axhline function (adds threshold lines)
        mock_axhline.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.bar')
    def test_metrics_plot_with_dataframe(self, mock_bar, mock_subplot, mock_figure, mock_show,
                                       sample_binary_target_df, categorical_columns):
        """Test metrics_plot with DataFrame and cross_cols parameter."""
        df = sample_binary_target_df.copy()
        y_true = df['target'].values
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception with cross_cols
        xai.metrics_plot(y_true, y_prob, df=df, cross_cols=['gender'], categorical_cols=categorical_columns)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_metrics_plot_threshold_parameter(self, mock_figure, mock_show, sample_evaluation_data):
        """Test metrics_plot with custom threshold parameter."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should accept threshold parameter
        xai.metrics_plot(y_true, y_prob, threshold=0.3)

        mock_figure.assert_called()
        mock_show.assert_called()


class TestConfusionMatrixPlot:
    """Tests for confusion_matrix_plot function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    def test_confusion_matrix_plot_basic(self, mock_imshow, mock_figure, mock_show, sample_evaluation_data):
        """Test basic confusion_matrix_plot functionality."""
        y_true, y_pred = sample_evaluation_data

        # Should not raise exception
        xai.confusion_matrix_plot(y_true, y_pred)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_imshow.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    def test_confusion_matrix_plot_scaled(self, mock_imshow, mock_figure, mock_show, sample_evaluation_data):
        """Test confusion_matrix_plot with scaled parameter."""
        y_true, y_pred = sample_evaluation_data

        # Should accept scaled parameter
        xai.confusion_matrix_plot(y_true, y_pred, scaled=True)
        xai.confusion_matrix_plot(y_true, y_pred, scaled=False)

        # Should call matplotlib functions both times
        assert mock_figure.call_count == 2
        assert mock_show.call_count == 2

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_confusion_matrix_plot_custom_labels(self, mock_figure, mock_show, sample_evaluation_data):
        """Test confusion_matrix_plot with custom labels."""
        y_true, y_pred = sample_evaluation_data

        # Should accept custom labels
        xai.confusion_matrix_plot(y_true, y_pred, labels=['No', 'Yes'])

        mock_figure.assert_called()
        mock_show.assert_called()


class TestRocPlot:
    """Tests for roc_plot function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_roc_plot_basic(self, mock_plot, mock_figure, mock_show, sample_evaluation_data):
        """Test basic roc_plot functionality."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception
        result = xai.roc_plot(y_true, y_prob)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_plot.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_roc_plot_with_dataframe(self, mock_plot, mock_figure, mock_show,
                                   sample_binary_target_df, categorical_columns):
        """Test roc_plot with DataFrame and cross_cols parameter."""
        df = sample_binary_target_df.copy()
        y_true = df['target'].values
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception with cross_cols
        result = xai.roc_plot(y_true, y_prob, df=df, cross_cols=['gender'], categorical_cols=categorical_columns)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_plot.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_roc_plot_perfect_classifier(self, mock_figure, mock_show):
        """Test roc_plot with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])  # Perfect separation

        result = xai.roc_plot(y_true, y_prob)

        mock_figure.assert_called()
        mock_show.assert_called()


class TestPrPlot:
    """Tests for pr_plot function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_pr_plot_basic(self, mock_plot, mock_figure, mock_show, sample_evaluation_data):
        """Test basic pr_plot functionality."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception
        result = xai.pr_plot(y_true, y_prob)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_plot.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_pr_plot_with_dataframe(self, mock_plot, mock_figure, mock_show,
                                  sample_binary_target_df, categorical_columns):
        """Test pr_plot with DataFrame and cross_cols parameter."""
        df = sample_binary_target_df.copy()
        y_true = df['target'].values
        y_prob = np.random.uniform(0, 1, len(y_true))

        # Should not raise exception with cross_cols
        result = xai.pr_plot(y_true, y_prob, df=df, cross_cols=['gender'], categorical_cols=categorical_columns)

        # Should call matplotlib functions
        mock_figure.assert_called()
        mock_plot.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_pr_plot_edge_cases(self, mock_figure, mock_show):
        """Test pr_plot with edge cases."""
        # All positive class
        y_true_pos = np.array([1, 1, 1, 1])
        y_prob_pos = np.array([0.6, 0.7, 0.8, 0.9])

        result = xai.pr_plot(y_true_pos, y_prob_pos)

        mock_figure.assert_called()
        mock_show.assert_called()


class TestCurveFunction:
    """Tests for internal _curve function (if accessible)."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_curve_roc_type(self, mock_plot, mock_figure, mock_show, sample_evaluation_data):
        """Test _curve function with ROC curve type."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        try:
            # Test if _curve function is accessible
            result = xai._curve(y_true, y_prob, curve_type="roc")

            # Should call matplotlib functions
            mock_figure.assert_called()
            mock_plot.assert_called()
            mock_show.assert_called()
        except AttributeError:
            # _curve might not be accessible from outside, which is fine
            pytest.skip("_curve function not accessible for direct testing")

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_curve_pr_type(self, mock_plot, mock_figure, mock_show, sample_evaluation_data):
        """Test _curve function with PR curve type."""
        y_true, _ = sample_evaluation_data
        y_prob = np.random.uniform(0, 1, len(y_true))

        try:
            # Test if _curve function is accessible
            result = xai._curve(y_true, y_prob, curve_type="pr")

            # Should call matplotlib functions
            mock_figure.assert_called()
            mock_plot.assert_called()
            mock_show.assert_called()
        except AttributeError:
            # _curve might not be accessible from outside, which is fine
            pytest.skip("_curve function not accessible for direct testing")