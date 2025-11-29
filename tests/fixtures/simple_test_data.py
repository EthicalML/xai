"""Simple, focused test data generators for XAI library tests.

Clean, minimal data generators designed for essential functionality testing.
No complex edge cases - just focused data for validating core behaviors.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def basic_mixed_data() -> pd.DataFrame:
    """
    Small DataFrame with mixed data types for general testing.
    50 rows with categorical, numeric, and target columns.
    """
    np.random.seed(42)  # Reproducible tests

    return pd.DataFrame({
        # Categorical columns
        'gender': np.random.choice(['Male', 'Female'], 50),
        'category': np.random.choice(['A', 'B', 'C'], 50),
        'boolean_col': np.random.choice([True, False], 50),

        # Numeric columns (with known statistical properties)
        'numeric_normal': np.random.normal(100, 15, 50),  # μ=100, σ=15
        'numeric_uniform': np.random.uniform(0, 1000, 50),
        'age': np.random.randint(18, 80, 50),

        # Target column
        'target': np.random.choice([0, 1], 50)
    })


def imbalanced_binary_data() -> pd.DataFrame:
    """
    DataFrame with known class imbalance for balance testing.
    80% class 0, 20% class 1 - clear imbalance to test rebalancing.
    """
    np.random.seed(42)

    # Create imbalanced target: 80 zeros, 20 ones
    target = [0] * 80 + [1] * 20
    np.random.shuffle(target)

    return pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], 100),
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': target
    })


def normalization_test_data() -> pd.DataFrame:
    """
    DataFrame with known statistical properties for normalization testing.
    Columns have different means and standard deviations.
    """
    np.random.seed(42)

    return pd.DataFrame({
        'col_mean_100_std_15': np.random.normal(100, 15, 50),
        'col_mean_0_std_1': np.random.normal(0, 1, 50),
        'col_mean_500_std_100': np.random.normal(500, 100, 50),
        'categorical': ['A', 'B', 'C'] * 16 + ['A', 'B'],  # Should be ignored
        'target': np.random.choice([0, 1], 50)
    })


def categorical_test_data() -> pd.DataFrame:
    """
    DataFrame focused on categorical data conversion testing.
    Mix of string, object, and boolean columns.
    """
    return pd.DataFrame({
        'string_cat': ['Cat1', 'Cat2', 'Cat3'] * 10,
        'object_cat': pd.Series(['X', 'Y', 'Z'] * 10, dtype='object'),
        'bool_cat': [True, False] * 15,
        'numeric_col': range(30),  # Should remain unchanged
    })


def evaluation_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Known test case for evaluation metrics with hand-calculated expected results.
    Returns (y_true, y_pred) for metrics validation.

    Expected metrics:
    - Accuracy: 0.6 (6/10 correct)
    - Precision: 0.75 (3/4 predicted positives are correct)
    - Recall: 0.6 (3/5 actual positives found)
    - F1: 0.667 (2 * 0.75 * 0.6 / (0.75 + 0.6))
    """
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
    return y_true, y_pred


def correlation_test_data() -> pd.DataFrame:
    """
    DataFrame with known correlation patterns for testing correlations function.
    """
    np.random.seed(42)

    # Create correlated data
    x = np.random.normal(0, 1, 50)
    y = 0.8 * x + 0.2 * np.random.normal(0, 1, 50)  # Strong positive correlation
    z = -0.6 * x + 0.4 * np.random.normal(0, 1, 50)  # Negative correlation

    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'independent': np.random.normal(0, 1, 50),  # Should have low correlation
        'category': np.random.choice(['A', 'B'], 50)
    })


def feature_importance_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    DataFrame for feature importance testing with known feature relationships.
    """
    np.random.seed(42)

    # Create features with different importance levels
    important_feature = np.random.normal(0, 1, 100)
    noise_feature = np.random.normal(0, 1, 100)

    # Target depends strongly on important_feature, weakly on noise_feature
    target = (important_feature > 0).astype(int) + 0.1 * (noise_feature > 0).astype(int)
    target = (target > 0.5).astype(int)

    X = pd.DataFrame({
        'important_feature': important_feature,
        'noise_feature': noise_feature,
        'random_feature': np.random.normal(0, 1, 100)
    })

    y = pd.Series(target)

    return X, y


def probability_bucketing_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Data for smile_imbalance probability bucketing tests.
    Returns (y_true, y_prob) with known bucket distributions.
    """
    # Create data where high probabilities correlate with positive class
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.2, 0.6, 0.8, 0.9, 0.7])  # Clear separation

    return y_true, y_prob.reshape(-1, 1)


# Helper functions for common column lists
def get_categorical_columns() -> list:
    """Standard categorical column names for testing."""
    return ['gender', 'category', 'boolean_col']


def get_numeric_columns() -> list:
    """Standard numeric column names for testing."""
    return ['numeric_normal', 'numeric_uniform', 'age']


def assert_statistical_properties(series: pd.Series, expected_mean: float = 0.0,
                                expected_std: float = 1.0, tolerance: float = 0.1):
    """
    Helper function to assert statistical properties (for normalization tests).
    """
    actual_mean = series.mean()
    actual_std = series.std()

    assert abs(actual_mean - expected_mean) < tolerance, \
        f"Mean {actual_mean} not within {tolerance} of expected {expected_mean}"
    assert abs(actual_std - expected_std) < tolerance, \
        f"Std {actual_std} not within {tolerance} of expected {expected_std}"