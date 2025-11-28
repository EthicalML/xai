"""Pytest configuration and shared fixtures for XAI library tests."""
import pytest
import pandas as pd
import numpy as np
import matplotlib
from unittest.mock import Mock, MagicMock
from typing import Any, Callable

# Configure matplotlib for headless testing
matplotlib.use('Agg')  # Use non-interactive backend for tests

from .fixtures.synthetic_data import (
    basic_df, categorical_df, numeric_df, binary_target_df,
    empty_df, single_row_df, missing_data_df,
    evaluation_test_data, prediction_probabilities,
    get_categorical_columns, get_numerical_columns
)


@pytest.fixture
def sample_basic_df() -> pd.DataFrame:
    """Fixture providing basic test DataFrame."""
    return basic_df()


@pytest.fixture
def sample_categorical_df() -> pd.DataFrame:
    """Fixture providing categorical test DataFrame."""
    return categorical_df()


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """Fixture providing numeric test DataFrame."""
    return numeric_df()


@pytest.fixture
def sample_binary_target_df() -> pd.DataFrame:
    """Fixture providing binary target test DataFrame."""
    return binary_target_df()


@pytest.fixture
def sample_empty_df() -> pd.DataFrame:
    """Fixture providing empty DataFrame."""
    return empty_df()


@pytest.fixture
def sample_single_row_df() -> pd.DataFrame:
    """Fixture providing single row DataFrame."""
    return single_row_df()


@pytest.fixture
def sample_missing_data_df() -> pd.DataFrame:
    """Fixture providing DataFrame with missing values."""
    return missing_data_df()


@pytest.fixture
def categorical_columns() -> list:
    """Fixture providing list of categorical column names."""
    return get_categorical_columns()


@pytest.fixture
def numerical_columns() -> list:
    """Fixture providing list of numerical column names."""
    return get_numerical_columns()


@pytest.fixture
def mock_sklearn_model():
    """Fixture providing mock sklearn model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    mock_model.evaluate.return_value = [0.4, 0.85]  # [loss, accuracy]
    return mock_model


@pytest.fixture
def mock_evaluation_function():
    """Fixture providing mock evaluation function for feature importance testing."""
    def mock_eval(x, y):
        """Mock evaluation function that returns consistent accuracy."""
        return 0.85  # Fixed accuracy for testing
    return mock_eval


@pytest.fixture
def sample_predictions():
    """Fixture providing sample prediction probabilities."""
    return prediction_probabilities()


@pytest.fixture
def sample_evaluation_data():
    """Fixture providing sample evaluation data (y_true, y_pred)."""
    return evaluation_test_data()


# Test utilities
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """
    Custom assertion for DataFrame comparison with better error messages.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtype: Whether to check data types (default: True)
    """
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_array_almost_equal(arr1: np.ndarray, arr2: np.ndarray, decimal: int = 7):
    """
    Custom assertion for numpy array comparison with tolerance.

    Args:
        arr1: First array
        arr2: Second array
        decimal: Number of decimal places for comparison (default: 7)
    """
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


def assert_statistical_properties(series: pd.Series, expected_mean: float = 0.0,
                                expected_std: float = 1.0, tolerance: float = 0.1):
    """
    Assert that a pandas Series has expected statistical properties.

    Args:
        series: Pandas Series to check
        expected_mean: Expected mean value (default: 0.0)
        expected_std: Expected standard deviation (default: 1.0)
        tolerance: Tolerance for comparison (default: 0.1)
    """
    actual_mean = series.mean()
    actual_std = series.std()

    assert abs(actual_mean - expected_mean) < tolerance, \
        f"Mean {actual_mean} not within {tolerance} of expected {expected_mean}"
    assert abs(actual_std - expected_std) < tolerance, \
        f"Std {actual_std} not within {tolerance} of expected {expected_std}"


def create_mock_matplotlib_patches():
    """Create a dictionary of matplotlib mocks for patching."""
    mocks = {
        'matplotlib.pyplot.show': Mock(),
        'matplotlib.pyplot.figure': Mock(),
        'matplotlib.pyplot.subplot': Mock(),
        'matplotlib.pyplot.plot': Mock(),
        'matplotlib.pyplot.bar': Mock(),
        'matplotlib.pyplot.scatter': Mock(),
        'matplotlib.pyplot.hist': Mock(),
        'matplotlib.pyplot.xlabel': Mock(),
        'matplotlib.pyplot.ylabel': Mock(),
        'matplotlib.pyplot.title': Mock(),
        'matplotlib.pyplot.legend': Mock(),
        'matplotlib.pyplot.tight_layout': Mock(),
        'matplotlib.pyplot.subplots': Mock(return_value=(Mock(), Mock())),
    }
    return mocks


class MockCSVFile:
    """Mock class for simulating CSV file operations in tests."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def read_csv(self, *args, **kwargs):
        """Mock pd.read_csv that returns the test data."""
        return self.data.copy()


# Global test configuration
def pytest_configure():
    """Global pytest configuration."""
    # Ensure matplotlib doesn't try to open windows during testing
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode