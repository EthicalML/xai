"""Essential tests for XAI data preprocessing functions.

Focuses on core functionality validation:
- normalize_numeric: Statistical correctness (μ=0, σ=1)
- convert_categories: Categorical encoding correctness
- group_by_columns: Grouping and binning logic
"""
import pandas as pd
import numpy as np
import xai
from .conftest import assert_dataframe_equal, assert_array_almost_equal
from .fixtures.simple_test_data import assert_statistical_properties


class TestNormalizeNumeric:
    """Essential tests for normalize_numeric function."""

    def test_normalization_correctness(self, normalization_data):
        """Test that normalization achieves μ=0, σ=1 for all numeric columns."""
        result = xai.normalize_numeric(normalization_data)

        # Verify normalization worked correctly
        assert_statistical_properties(result['col_mean_100_std_15'], 0.0, 1.0)
        assert_statistical_properties(result['col_mean_0_std_1'], 0.0, 1.0)
        assert_statistical_properties(result['col_mean_500_std_100'], 0.0, 1.0)

        # Categorical column should be unchanged
        assert result['categorical'].equals(normalization_data['categorical'])

    def test_explicit_column_selection(self, normalization_data):
        """Test normalization with explicitly specified columns."""
        # Only normalize one specific column
        result = xai.normalize_numeric(normalization_data,
                                     numerical_cols=['col_mean_100_std_15'])

        # Only specified column should be normalized
        assert_statistical_properties(result['col_mean_100_std_15'], 0.0, 1.0)

        # Other numeric columns should remain unchanged
        assert result['col_mean_0_std_1'].equals(normalization_data['col_mean_0_std_1'])
        assert result['col_mean_500_std_100'].equals(normalization_data['col_mean_500_std_100'])

    def test_data_types_after_normalization(self, normalization_data):
        """Test that normalized columns have correct dtype."""
        result = xai.normalize_numeric(normalization_data)

        # Normalized columns should be float32
        numeric_columns = ['col_mean_100_std_15', 'col_mean_0_std_1', 'col_mean_500_std_100']
        for col in numeric_columns:
            assert result[col].dtype == np.float32

    def test_original_data_unchanged(self, normalization_data):
        """Test that original DataFrame is not modified."""
        original = normalization_data.copy()
        xai.normalize_numeric(normalization_data)
        assert_dataframe_equal(normalization_data, original)


class TestConvertCategories:
    """Essential tests for convert_categories function."""

    def test_categorical_conversion_correctness(self, categorical_data):
        """Test that categorical conversion produces correct int8 codes."""
        result = xai.convert_categories(categorical_data)

        # String and object columns should be converted to int8
        assert result['string_cat'].dtype == np.int8
        assert result['object_cat'].dtype == np.int8
        assert result['bool_cat'].dtype == np.int8

        # Numeric column should remain unchanged
        assert result['numeric_col'].equals(categorical_data['numeric_col'])

    def test_categorical_codes_mapping(self, categorical_data):
        """Test that categories map to sequential integers consistently."""
        result = xai.convert_categories(categorical_data,
                                      categorical_cols=['string_cat'])

        # Categories should be mapped to sequential integers (0, 1, 2)
        unique_codes = set(result['string_cat'].unique())
        expected_codes = {0, 1, 2}  # For Cat1, Cat2, Cat3
        assert unique_codes == expected_codes

    def test_boolean_conversion(self):
        """Test that boolean values convert to consistent integer mapping."""
        df = pd.DataFrame({'bool_col': [True, False, True, False]})
        result = xai.convert_categories(df)

        assert result['bool_col'].dtype == np.int8
        # True/False should map to consistent values (0, 1)
        unique_values = set(result['bool_col'].unique())
        assert unique_values.issubset({0, 1})

    def test_explicit_column_selection(self, categorical_data):
        """Test conversion with only specified columns."""
        result = xai.convert_categories(categorical_data,
                                      categorical_cols=['string_cat'])

        # Only specified column should be converted
        assert result['string_cat'].dtype == np.int8

        # Other categorical columns should remain unchanged
        original_types = categorical_data.dtypes
        assert result['bool_cat'].dtype == original_types['bool_cat']
        assert result['object_cat'].dtype == original_types['object_cat']

    def test_original_data_unchanged(self, categorical_data):
        """Test that original DataFrame is not modified."""
        original = categorical_data.copy()
        xai.convert_categories(categorical_data)
        assert_dataframe_equal(categorical_data, original)


class TestGroupByColumns:
    """Essential tests for group_by_columns function."""

    def test_categorical_grouping(self, mixed_data, categorical_columns):
        """Test grouping by categorical columns."""
        groups = list(xai.group_by_columns(mixed_data, ['gender'],
                                          categorical_cols=categorical_columns))

        # Should return groups for each unique gender value
        assert len(groups) >= 1  # At least one group

        for group_key, group_df in groups:
            # Each group should have proper structure
            assert isinstance(group_key, tuple)
            assert isinstance(group_df, pd.DataFrame)
            assert len(group_df) > 0
            assert len(group_key) == 1  # One grouping column

    def test_numeric_binning(self, mixed_data, categorical_columns):
        """Test grouping numeric column with binning."""
        groups = list(xai.group_by_columns(mixed_data, ['age'], bins=3,
                                          categorical_cols=categorical_columns))

        # Should create bins for the age column
        assert len(groups) <= 3  # Should not exceed specified number of bins
        assert len(groups) >= 1  # Should have at least one group

        for group_key, group_df in groups:
            assert isinstance(group_key, tuple)
            assert isinstance(group_df, pd.DataFrame)
            assert len(group_df) > 0

    def test_multiple_column_grouping(self, mixed_data, categorical_columns):
        """Test grouping by multiple columns."""
        groups = list(xai.group_by_columns(mixed_data, ['gender', 'category'],
                                          categorical_cols=categorical_columns))

        # Should return groups for combinations of gender and category
        for group_key, group_df in groups:
            assert len(group_key) == 2  # Two grouping columns
            assert isinstance(group_df, pd.DataFrame)
            assert len(group_df) > 0

    def test_generator_behavior(self, mixed_data, categorical_columns):
        """Test that function returns an iterable generator."""
        result = xai.group_by_columns(mixed_data, ['gender'],
                                     categorical_cols=categorical_columns)

        # Should be iterable
        assert hasattr(result, '__iter__')

        # Should be able to iterate multiple times by calling function again
        groups1 = list(xai.group_by_columns(mixed_data, ['gender'],
                                           categorical_cols=categorical_columns))
        groups2 = list(xai.group_by_columns(mixed_data, ['gender'],
                                           categorical_cols=categorical_columns))

        assert len(groups1) == len(groups2)

    def test_custom_bins(self, mixed_data, categorical_columns):
        """Test grouping with different bin numbers."""
        groups_3_bins = list(xai.group_by_columns(mixed_data, ['age'], bins=3,
                                                 categorical_cols=categorical_columns))
        groups_5_bins = list(xai.group_by_columns(mixed_data, ['age'], bins=5,
                                                 categorical_cols=categorical_columns))

        # Different bin numbers should potentially create different group counts
        assert isinstance(groups_3_bins, list)
        assert isinstance(groups_5_bins, list)
        assert len(groups_3_bins) <= 3
        assert len(groups_5_bins) <= 5

    def test_original_data_unchanged(self, mixed_data, categorical_columns):
        """Test that original DataFrame is not modified."""
        original = mixed_data.copy()
        list(xai.group_by_columns(mixed_data, ['gender'],
                                 categorical_cols=categorical_columns))
        assert_dataframe_equal(mixed_data, original)