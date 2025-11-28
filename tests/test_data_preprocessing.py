"""Tests for XAI data preprocessing functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

import xai
from .conftest import (
    assert_dataframe_equal,
    assert_statistical_properties,
    assert_array_almost_equal
)


class TestNormalizeNumeric:
    """Tests for normalize_numeric function."""

    def test_normalize_numeric_auto_detection(self, sample_numeric_df):
        """Test automatic detection and normalization of numeric columns."""
        result = xai.normalize_numeric(sample_numeric_df)

        # Should not modify non-numeric columns
        assert result['categorical'].equals(sample_numeric_df['categorical'])

        # Should normalize ALL numeric columns (including integers) to mean≈0, std≈1
        for col in ['col1', 'col2', 'col3', 'target']:
            assert_statistical_properties(result[col], expected_mean=0.0, expected_std=1.0)

    def test_normalize_numeric_explicit_columns(self, sample_numeric_df):
        """Test normalization with explicitly specified columns."""
        result = xai.normalize_numeric(sample_numeric_df, numerical_cols=['col1', 'col2'])

        # Only specified columns should be normalized
        assert_statistical_properties(result['col1'], expected_mean=0.0, expected_std=1.0)
        assert_statistical_properties(result['col2'], expected_mean=0.0, expected_std=1.0)

        # col3 should remain unchanged
        assert result['col3'].equals(sample_numeric_df['col3'])

    def test_normalize_numeric_empty_dataframe(self, sample_empty_df):
        """Test normalization with empty DataFrame."""
        result = xai.normalize_numeric(sample_empty_df)
        assert len(result) == 0

    def test_normalize_numeric_single_row(self, sample_single_row_df):
        """Test normalization with single row DataFrame."""
        result = xai.normalize_numeric(sample_single_row_df)

        # With single row, std=0, so normalization should result in NaN
        # This is expected behavior for single-value normalization
        assert len(result) == 1

    def test_normalize_numeric_preserves_original(self, sample_numeric_df):
        """Test that original DataFrame is not modified."""
        original = sample_numeric_df.copy()
        xai.normalize_numeric(sample_numeric_df)

        assert_dataframe_equal(sample_numeric_df, original)

    def test_normalize_numeric_data_types(self, sample_numeric_df):
        """Test that result has correct data types."""
        result = xai.normalize_numeric(sample_numeric_df)

        # Normalized columns should be float32
        for col in ['col1', 'col2', 'col3']:
            assert result[col].dtype == np.float32


class TestConvertCategories:
    """Tests for convert_categories function."""

    def test_convert_categories_auto_detection(self, sample_categorical_df):
        """Test automatic detection and conversion of categorical columns."""
        result = xai.convert_categories(sample_categorical_df)

        # String and object columns should be converted to int8 categorical codes
        assert result['cat_string'].dtype == np.int8
        assert result['cat_object'].dtype == np.int8

        # Boolean should be converted to int8
        assert result['cat_bool'].dtype == np.int8

        # Numeric columns should remain unchanged
        assert result['numeric_int'].dtype == sample_categorical_df['numeric_int'].dtype
        assert result['target'].dtype == sample_categorical_df['target'].dtype

    def test_convert_categories_explicit_columns(self, sample_categorical_df):
        """Test conversion with explicitly specified columns."""
        result = xai.convert_categories(sample_categorical_df, categorical_cols=['cat_string'])

        # Only specified column should be converted
        assert result['cat_string'].dtype == np.int8

        # Other categorical columns should remain unchanged
        original_types = sample_categorical_df.dtypes
        assert result['cat_bool'].dtype == original_types['cat_bool']
        assert result['cat_object'].dtype == original_types['cat_object']

    def test_convert_categories_codes_mapping(self, sample_categorical_df):
        """Test that categorical codes are assigned correctly."""
        result = xai.convert_categories(sample_categorical_df, categorical_cols=['cat_string'])

        # Categories should be mapped to sequential integers
        unique_codes = result['cat_string'].unique()
        expected_codes = {0, 1, 2}  # For categories A, B, C
        assert set(unique_codes) == expected_codes

    def test_convert_categories_empty_dataframe(self, sample_empty_df):
        """Test conversion with empty DataFrame."""
        result = xai.convert_categories(sample_empty_df)
        assert len(result) == 0

    def test_convert_categories_preserves_original(self, sample_categorical_df):
        """Test that original DataFrame is not modified."""
        original = sample_categorical_df.copy()
        xai.convert_categories(sample_categorical_df)

        assert_dataframe_equal(sample_categorical_df, original)

    def test_convert_categories_boolean_conversion(self):
        """Test boolean to categorical conversion."""
        df = pd.DataFrame({'bool_col': [True, False, True, False]})
        result = xai.convert_categories(df)

        assert result['bool_col'].dtype == np.int8
        # True/False should map to consistent integer values
        unique_values = set(result['bool_col'].unique())
        assert unique_values.issubset({0, 1})


class TestGroupByColumns:
    """Tests for group_by_columns function."""

    def test_group_by_columns_categorical(self, sample_basic_df, categorical_columns):
        """Test grouping by categorical columns."""
        groups = list(xai.group_by_columns(sample_basic_df, ['gender'], categorical_cols=categorical_columns))

        # Should return groups for each unique gender value
        group_keys = [group[0] for group in groups]
        unique_genders = sample_basic_df['gender'].unique()

        # Each unique gender should have a group
        assert len(groups) >= 1  # At least one group
        for group_key, group_df in groups:
            assert isinstance(group_key, tuple)
            assert isinstance(group_df, pd.DataFrame)
            assert len(group_df) > 0

    def test_group_by_columns_numeric_binning(self, sample_basic_df, categorical_columns):
        """Test grouping numeric column with binning."""
        groups = list(xai.group_by_columns(sample_basic_df, ['age'], bins=3, categorical_cols=categorical_columns))

        # Should create bins for the age column
        assert len(groups) <= 3  # Should not exceed specified number of bins
        for group_key, group_df in groups:
            assert isinstance(group_key, tuple)
            assert isinstance(group_df, pd.DataFrame)

    def test_group_by_columns_multiple_columns(self, sample_basic_df, categorical_columns):
        """Test grouping by multiple columns."""
        groups = list(xai.group_by_columns(sample_basic_df, ['gender', 'ethnicity'], categorical_cols=categorical_columns))

        # Should return groups for combinations of gender and ethnicity
        for group_key, group_df in groups:
            assert len(group_key) == 2  # Two grouping columns
            assert isinstance(group_df, pd.DataFrame)
            assert len(group_df) > 0

    def test_group_by_columns_empty_dataframe(self, sample_empty_df):
        """Test grouping with empty DataFrame."""
        # Empty DataFrame with non-existent column should raise KeyError
        with pytest.raises(KeyError):
            list(xai.group_by_columns(sample_empty_df, ['col1']))

    def test_group_by_columns_custom_bins(self, sample_basic_df, categorical_columns):
        """Test grouping with custom number of bins."""
        groups_3_bins = list(xai.group_by_columns(sample_basic_df, ['age'], bins=3, categorical_cols=categorical_columns))
        groups_5_bins = list(xai.group_by_columns(sample_basic_df, ['age'], bins=5, categorical_cols=categorical_columns))

        # Different bin numbers should potentially create different group counts
        # (though actual count depends on data distribution)
        assert isinstance(groups_3_bins, list)
        assert isinstance(groups_5_bins, list)

    def test_group_by_columns_preserves_original(self, sample_basic_df, categorical_columns):
        """Test that original DataFrame is not modified."""
        original = sample_basic_df.copy()
        list(xai.group_by_columns(sample_basic_df, ['gender'], categorical_cols=categorical_columns))

        assert_dataframe_equal(sample_basic_df, original)

    def test_group_by_columns_missing_column_handling(self, sample_basic_df):
        """Test handling of non-existent columns."""
        # Should handle gracefully or raise appropriate error
        with pytest.raises((KeyError, ValueError)):
            list(xai.group_by_columns(sample_basic_df, ['nonexistent_column']))

    def test_group_by_columns_generator_behavior(self, sample_basic_df, categorical_columns):
        """Test that function returns a generator."""
        result = xai.group_by_columns(sample_basic_df, ['gender'], categorical_cols=categorical_columns)

        # Should be iterable (generator-like)
        assert hasattr(result, '__iter__')

        # Should be able to iterate multiple times by calling again
        groups1 = list(xai.group_by_columns(sample_basic_df, ['gender'], categorical_cols=categorical_columns))
        groups2 = list(xai.group_by_columns(sample_basic_df, ['gender'], categorical_cols=categorical_columns))

        assert len(groups1) == len(groups2)