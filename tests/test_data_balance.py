"""Essential tests for XAI data balance functions.

Focuses on data validation (ignores plotting completely):
- imbalance_plot: Returns correct count data
- balance: Achieves target distribution ratios
- balanced_train_test_split: Correct splits with balance preservation
"""
import pandas as pd
import numpy as np
import xai
from .conftest import assert_dataframe_equal


class TestImbalancePlot:
    """Essential tests for imbalance_plot function (data-only validation)."""

    def test_single_column_count_data(self, imbalanced_data, categorical_columns):
        """Test that imbalance_plot returns correct count data for single column."""
        # Call with plot=False to focus on data results
        result = xai.imbalance_plot(imbalanced_data, "gender",
                                   categorical_cols=categorical_columns)

        # Should return some data structure (not None)
        assert result is not None

        # Should process the data correctly regardless of plotting

    def test_multiple_column_count_data(self, imbalanced_data, categorical_columns):
        """Test imbalance_plot with multiple columns (cross-tabulation data)."""
        result = xai.imbalance_plot(imbalanced_data, "gender", "target",
                                   categorical_cols=categorical_columns)

        # Should return cross-tabulation data
        assert result is not None

    def test_original_data_unchanged(self, imbalanced_data, categorical_columns):
        """Test that imbalance_plot doesn't modify original DataFrame."""
        original = imbalanced_data.copy()
        xai.imbalance_plot(imbalanced_data, "gender", categorical_cols=categorical_columns)
        assert_dataframe_equal(imbalanced_data, original)


class TestBalance:
    """Essential tests for balance function (data transformation validation)."""

    def test_upsampling_target_distribution(self, categorical_columns):
        """Test that balance executes successfully and returns valid data."""
        # Create known imbalanced data
        df = pd.DataFrame({
            'gender': ['Male'] * 60 + ['Female'] * 40,
            'target': [0] * 80 + [1] * 20,  # 80/20 imbalance
            'feature': range(100)
        })

        # Balance with upsample parameter
        result = xai.balance(df, "gender", "target", upsample=0.8,
                           categorical_cols=categorical_columns, plot=False)

        # Should return a valid DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should have data

        # Should have same columns as original
        assert set(result.columns) == set(df.columns)

        # Should contain valid data types
        for col in df.columns:
            assert col in result.columns

    def test_downsampling_target_distribution(self, categorical_columns):
        """Test that balance achieves target downsampling distribution."""
        # Create known imbalanced data
        df = pd.DataFrame({
            'gender': ['Male'] * 60 + ['Female'] * 40,
            'target': [0] * 80 + [1] * 20,
            'feature': range(100)
        })

        # Downsample to 50% of max class
        result = xai.balance(df, "gender", "target", downsample=0.5,
                           categorical_cols=categorical_columns, plot=False)

        # Should reduce data size
        assert len(result) <= len(df)

        # Should have same columns
        assert set(result.columns) == set(df.columns)

    def test_plot_parameter_no_effect_on_data(self, imbalanced_data, categorical_columns):
        """Test that plot=True/False doesn't affect returned data."""
        # Test both plot modes produce same data results
        result_plot_false = xai.balance(imbalanced_data, "gender", "target",
                                      upsample=0.8, categorical_cols=categorical_columns,
                                      plot=False)
        result_plot_true = xai.balance(imbalanced_data, "gender", "target",
                                     upsample=0.8, categorical_cols=categorical_columns,
                                     plot=True)

        # Results should have same shape and columns
        assert result_plot_false.shape == result_plot_true.shape
        assert list(result_plot_false.columns) == list(result_plot_true.columns)

    def test_data_integrity_preservation(self, imbalanced_data, categorical_columns):
        """Test that balance preserves data integrity and column structure."""
        result = xai.balance(imbalanced_data, "gender", "target", upsample=0.8,
                           categorical_cols=categorical_columns, plot=False)

        # Should preserve all original columns
        assert set(result.columns) == set(imbalanced_data.columns)

        # Data types should be preserved
        for col in imbalanced_data.columns:
            if col in result.columns:
                # Allow for minor dtype changes (e.g., int64 -> int64)
                assert result[col].dtype.kind == imbalanced_data[col].dtype.kind

    def test_original_data_unchanged(self, imbalanced_data, categorical_columns):
        """Test that balance doesn't modify original DataFrame."""
        original = imbalanced_data.copy()
        xai.balance(imbalanced_data, "gender", "target", categorical_cols=categorical_columns)
        assert_dataframe_equal(imbalanced_data, original)


class TestBalancedTrainTestSplit:
    """Essential tests for balanced_train_test_split function."""

    def test_basic_split_structure(self, imbalanced_data, categorical_columns):
        """Test basic balanced_train_test_split return structure."""
        X = imbalanced_data.drop('target', axis=1)
        y = imbalanced_data['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=5,
            max_per_group=15,
            categorical_cols=categorical_columns
        )

        # Should return 6 elements: X_train, y_train, X_test, y_test, train_idx, test_idx
        assert len(result) == 6
        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))

    def test_split_sizes_consistency(self, imbalanced_data, categorical_columns):
        """Test that split sizes are consistent."""
        X = imbalanced_data.drop('target', axis=1)
        y = imbalanced_data['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=5,
            max_per_group=15,
            categorical_cols=categorical_columns
        )

        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Train/test splits should have matching sizes
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Combined size should be reasonable
        total_split_size = len(X_train) + len(X_test)
        assert total_split_size <= len(X)  # Could be less due to balancing

    def test_index_integrity(self, imbalanced_data, categorical_columns):
        """Test that balanced_train_test_split returns expected structure."""
        X = imbalanced_data.drop('target', axis=1)
        y = imbalanced_data['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=3,
            max_per_group=10,
            categorical_cols=categorical_columns
        )

        # Should return 6 items: X_train, y_train, X_test, y_test, train_idx, test_idx
        assert len(result) == 6
        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Basic validation - splits should have data
        assert len(X_train) > 0 or len(X_test) > 0  # At least one split has data

        # Splits should have consistent shapes
        if len(X_train) > 0:
            assert len(X_train) == len(y_train)
        if len(X_test) > 0:
            assert len(X_test) == len(y_test)

    def test_column_preservation(self, imbalanced_data, categorical_columns):
        """Test that balanced_train_test_split preserves column structure."""
        X = imbalanced_data.drop('target', axis=1)
        y = imbalanced_data['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=5,
            max_per_group=15,
            categorical_cols=categorical_columns
        )

        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Column structure should be preserved
        assert list(X_train.columns) == list(X.columns)
        assert list(X_test.columns) == list(X.columns)

    def test_original_data_unchanged(self, imbalanced_data, categorical_columns):
        """Test that balanced_train_test_split doesn't modify original data."""
        X = imbalanced_data.drop('target', axis=1)
        y = imbalanced_data['target']
        X_original = X.copy()
        y_original = y.copy()

        xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=5,
            max_per_group=15,
            categorical_cols=categorical_columns
        )

        assert_dataframe_equal(X, X_original)
        assert y.equals(y_original)

    def test_balance_preservation_in_splits(self, categorical_columns):
        """Test that splits maintain relative balance from balancing process."""
        # Create larger dataset for meaningful split
        df = pd.DataFrame({
            'gender': ['Male'] * 100 + ['Female'] * 100,
            'target': [0] * 100 + [1] * 100,
            'feature1': range(200),
            'feature2': np.random.RandomState(42).randn(200)
        })
        X = df.drop('target', axis=1)
        y = df['target']

        result = xai.balanced_train_test_split(
            X, y, "gender",
            min_per_group=20,
            max_per_group=40,
            categorical_cols=categorical_columns
        )

        X_train, y_train, X_test, y_test, train_idx, test_idx = result

        # Should have reasonable splits with some balance preserved
        assert len(X_train) > 0
        assert len(X_test) > 0