"""Synthetic test data generators for XAI library tests."""
import pandas as pd
import numpy as np
from typing import List, Tuple


def basic_df() -> pd.DataFrame:
    """
    Small DataFrame (50 rows) with mixed data types mimicking census structure.
    Includes categorical, numeric, and binary target columns.
    """
    np.random.seed(42)  # For reproducible tests

    data = {
        'age': np.random.randint(18, 80, 50),
        'workclass': np.random.choice(['Private', 'Gov', 'Self-emp'], 50),
        'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters'], 50),
        'education-num': np.random.randint(1, 16, 50),
        'marital-status': np.random.choice(['Married', 'Single', 'Divorced'], 50),
        'occupation': np.random.choice(['Tech', 'Sales', 'Service'], 50),
        'relationship': np.random.choice(['Husband', 'Wife', 'Child'], 50),
        'ethnicity': np.random.choice(['White', 'Black', 'Asian'], 50),
        'gender': np.random.choice(['Male', 'Female'], 50),
        'capital-gain': np.random.randint(0, 5000, 50),
        'capital-loss': np.random.randint(0, 1000, 50),
        'hours-per-week': np.random.randint(20, 60, 50),
        'loan': np.random.choice(['<=50K', '>50K'], 50)
    }

    return pd.DataFrame(data)


def categorical_df() -> pd.DataFrame:
    """DataFrame focused on categorical columns for category conversion testing."""
    np.random.seed(42)

    data = {
        'cat_string': ['A', 'B', 'C', 'A', 'B'] * 10,
        'cat_bool': [True, False] * 25,
        'cat_object': pd.Series(['X', 'Y', 'Z'] * 16 + ['X', 'Y'], dtype='object'),
        'numeric_int': range(50),
        'target': np.random.choice([0, 1], 50)
    }

    return pd.DataFrame(data)


def numeric_df() -> pd.DataFrame:
    """DataFrame focused on numeric columns for normalization testing."""
    np.random.seed(42)

    data = {
        'col1': np.random.normal(100, 15, 50),  # Mean=100, std=15
        'col2': np.random.uniform(0, 1000, 50),  # Uniform distribution
        'col3': np.random.exponential(2, 50),    # Exponential distribution
        'categorical': ['A', 'B'] * 25,          # Should be ignored
        'target': np.random.choice([0, 1], 50)
    }

    return pd.DataFrame(data)


def binary_target_df() -> pd.DataFrame:
    """DataFrame with binary classification target for balance/evaluation testing."""
    np.random.seed(42)

    # Create imbalanced dataset for balance testing
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'gender': ['Male'] * 60 + ['Female'] * 40,  # Gender imbalance
        'ethnicity': ['White'] * 70 + ['Black'] * 20 + ['Asian'] * 10,  # Ethnic imbalance
        'target': [0] * 80 + [1] * 20  # 80/20 class imbalance
    }

    return pd.DataFrame(data)


def empty_df() -> pd.DataFrame:
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


def single_row_df() -> pd.DataFrame:
    """Single row DataFrame for minimal data edge case."""
    data = {
        'age': [25],
        'gender': ['Male'],
        'target': [1]
    }
    return pd.DataFrame(data)


def missing_data_df() -> pd.DataFrame:
    """DataFrame with NaN values for missing data testing."""
    np.random.seed(42)

    data = {
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['A', np.nan, 'C', 'D', 'E'],
        'col3': [1.1, 2.2, 3.3, np.nan, 5.5],
        'target': [0, 1, 0, np.nan, 1]
    }

    return pd.DataFrame(data)


def evaluation_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for evaluation metrics testing.
    Returns (y_true, y_pred) with known statistical properties.
    """
    # Create known test case for metric validation
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

    return y_true, y_pred


def prediction_probabilities() -> np.ndarray:
    """Generate test prediction probabilities for ROC/PR curve testing."""
    np.random.seed(42)
    return np.random.uniform(0, 1, 100).reshape(-1, 1)


def get_categorical_columns() -> List[str]:
    """Return list of categorical column names for testing."""
    return ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'ethnicity', 'gender', 'loan']


def get_numerical_columns() -> List[str]:
    """Return list of numerical column names for testing."""
    return ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']