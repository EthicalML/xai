"""Streamlined pytest configuration and fixtures for XAI library tests.

Essential fixtures only - focused on data validation without plotting complexity.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib

# Set non-interactive backend globally - no plot testing
matplotlib.use('Agg')

from .fixtures.simple_test_data import (
    basic_mixed_data,
    imbalanced_binary_data,
    normalization_test_data,
    categorical_test_data,
    evaluation_test_data,
    correlation_test_data,
    feature_importance_data as feature_importance_data_func,
    probability_bucketing_data,
    get_categorical_columns,
    get_numeric_columns,
    assert_statistical_properties
)


# Essential fixtures using our clean data generators
@pytest.fixture
def mixed_data():
    """Basic mixed data for general testing."""
    return basic_mixed_data()


@pytest.fixture
def imbalanced_data():
    """Known imbalanced data for balance testing."""
    return imbalanced_binary_data()


@pytest.fixture
def normalization_data():
    """Data with known statistical properties for normalization tests."""
    return normalization_test_data()


@pytest.fixture
def categorical_data():
    """Data focused on categorical conversion testing."""
    return categorical_test_data()


@pytest.fixture
def evaluation_data():
    """Known evaluation test case with hand-calculated metrics."""
    return evaluation_test_data()


@pytest.fixture
def correlation_data():
    """Data with known correlation patterns."""
    return correlation_test_data()


@pytest.fixture
def feature_importance_data():
    """Data for feature importance testing."""
    return feature_importance_data_func()


@pytest.fixture
def probability_data():
    """Data for probability bucketing tests."""
    return probability_bucketing_data()


@pytest.fixture
def categorical_columns():
    """Standard categorical column names."""
    return get_categorical_columns()


@pytest.fixture
def numeric_columns():
    """Standard numeric column names."""
    return get_numeric_columns()


# Essential helper functions
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """Assert DataFrame equality with clear error messages."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_array_almost_equal(arr1: np.ndarray, arr2: np.ndarray, decimal: int = 7):
    """Assert numpy array equality with tolerance."""
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


# Global test configuration
def pytest_configure():
    """Global pytest configuration - ensure matplotlib doesn't open windows."""
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode