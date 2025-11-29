"""Tests for XAI data loading functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import os

import xai.data
from .conftest import assert_dataframe_equal


class TestLoadCensus:
    """Tests for load_census function."""

    @pytest.fixture
    def mock_census_data(self):
        """Create mock census data for testing."""
        data = {
            'age': [25, 30, 35, 40],
            'workclass': ['Private', 'Gov', 'Self-emp', 'Private'],
            'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad'],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'loan': ['<=50K', '>50K', '>50K', '<=50K']
        }
        return pd.DataFrame(data)

    @patch('pandas.read_csv')
    def test_load_census_default_return(self, mock_read_csv, mock_census_data):
        """Test load_census with default return format (full DataFrame)."""
        mock_read_csv.return_value = mock_census_data

        result = xai.data.load_census()

        # Should call pd.read_csv with correct parameters
        mock_read_csv.assert_called_once()
        call_args = mock_read_csv.call_args

        # Should include index_col=0 parameter
        assert 'index_col' in call_args.kwargs
        assert call_args.kwargs['index_col'] == 0

        # Should return full DataFrame
        assert isinstance(result, pd.DataFrame)
        assert_dataframe_equal(result, mock_census_data)

    @patch('pandas.read_csv')
    def test_load_census_return_xy_false(self, mock_read_csv, mock_census_data):
        """Test load_census with return_xy=False (explicit)."""
        mock_read_csv.return_value = mock_census_data

        result = xai.data.load_census(return_xy=False)

        # Should return full DataFrame
        assert isinstance(result, pd.DataFrame)
        assert_dataframe_equal(result, mock_census_data)

    @patch('pandas.read_csv')
    def test_load_census_return_xy_true(self, mock_read_csv, mock_census_data):
        """Test load_census with return_xy=True (X, y split)."""
        mock_read_csv.return_value = mock_census_data

        result = xai.data.load_census(return_xy=True)

        # Should return tuple (X, y)
        assert isinstance(result, tuple)
        assert len(result) == 2

        X, y = result

        # X should be DataFrame without 'loan' column
        assert isinstance(X, pd.DataFrame)
        assert 'loan' not in X.columns
        expected_x_columns = ['age', 'workclass', 'education', 'gender']
        assert list(X.columns) == expected_x_columns

        # y should be Series with 'loan' column data
        assert isinstance(y, pd.Series)
        assert y.name == 'loan'
        expected_y_values = ['<=50K', '>50K', '>50K', '<=50K']
        assert list(y.values) == expected_y_values

    @patch('pandas.read_csv')
    def test_load_census_file_path_construction(self, mock_read_csv, mock_census_data):
        """Test that load_census constructs correct file path."""
        mock_read_csv.return_value = mock_census_data

        xai.data.load_census()

        # Should call read_csv with path including 'census.csv'
        call_args = mock_read_csv.call_args
        file_path = call_args[0][0]
        assert file_path.endswith('census.csv')
        assert 'xai/data' in file_path or 'xai\\data' in file_path  # Handle Windows paths

    @patch('pandas.read_csv')
    def test_load_census_missing_loan_column(self, mock_read_csv):
        """Test load_census behavior when 'loan' column is missing."""
        # Create data without 'loan' column
        data_without_loan = pd.DataFrame({
            'age': [25, 30],
            'gender': ['Male', 'Female']
        })
        mock_read_csv.return_value = data_without_loan

        # Default return should work fine
        result = xai.data.load_census(return_xy=False)
        assert isinstance(result, pd.DataFrame)

        # return_xy=True should raise KeyError for missing 'loan' column
        with pytest.raises(KeyError):
            xai.data.load_census(return_xy=True)

    @patch('pandas.read_csv')
    def test_load_census_empty_dataframe(self, mock_read_csv):
        """Test load_census with empty DataFrame."""
        empty_df = pd.DataFrame()
        mock_read_csv.return_value = empty_df

        # Default return should work
        result = xai.data.load_census(return_xy=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

        # return_xy=True should raise KeyError for missing 'loan' column
        with pytest.raises(KeyError):
            xai.data.load_census(return_xy=True)

    @patch('pandas.read_csv')
    def test_load_census_single_row(self, mock_read_csv):
        """Test load_census with single row DataFrame."""
        single_row_df = pd.DataFrame({
            'age': [25],
            'gender': ['Male'],
            'loan': ['<=50K']
        })
        mock_read_csv.return_value = single_row_df

        # Test return_xy=True with single row
        X, y = xai.data.load_census(return_xy=True)

        assert len(X) == 1
        assert len(y) == 1
        assert 'loan' not in X.columns
        assert y.iloc[0] == '<=50K'


    @patch('pandas.read_csv')
    def test_load_census_preserves_data_types(self, mock_read_csv, mock_census_data):
        """Test that load_census preserves original data types."""
        # Ensure mock data has specific dtypes
        mock_census_data = mock_census_data.astype({
            'age': 'int64',
            'workclass': 'object',
            'education': 'object',
            'gender': 'object',
            'loan': 'object'
        })
        mock_read_csv.return_value = mock_census_data

        result = xai.data.load_census()

        # Should preserve dtypes from CSV
        assert result['age'].dtype == mock_census_data['age'].dtype
        assert result['workclass'].dtype == mock_census_data['workclass'].dtype

    @patch('pandas.read_csv')
    def test_load_census_handles_index_correctly(self, mock_read_csv, mock_census_data):
        """Test that load_census handles index parameter correctly."""
        mock_read_csv.return_value = mock_census_data

        xai.data.load_census()

        # Should pass index_col=0 to read_csv
        mock_read_csv.assert_called_once()
        call_kwargs = mock_read_csv.call_args.kwargs
        assert 'index_col' in call_kwargs
        assert call_kwargs['index_col'] == 0