import scipy.stats
import numpy as np
import pandas as pd


def test_column_presence_and_type(data: pd.DataFrame):
    
    required_columns = {
        "id": pd.api.types.is_integer_dtype,
        "name": pd.api.types.is_object_dtype,
        "host_id": pd.api.types.is_integer_dtype,
        "host_name": pd.api.types.is_object_dtype,
        "neighbourhood_group": pd.api.types.is_object_dtype,
        "neighbourhood": pd.api.types.is_object_dtype,
        "latitude": pd.api.types.is_float_dtype,
        "longitude": pd.api.types.is_float_dtype,
        "room_type": pd.api.types.is_object_dtype,
        "price": pd.api.types.is_integer_dtype,
        "minimum_nights": pd.api.types.is_integer_dtype,
        "number_of_reviews": pd.api.types.is_integer_dtype,
        "last_review": pd.api.types.is_object_dtype,
        "reviews_per_month": pd.api.types.is_float_dtype,
        "calculated_host_listings_count": pd.api.types.is_integer_dtype,
        "availability_365": pd.api.types.is_integer_dtype
    }
    
    # Check column presence
    assert set(data.columns.values).issuperset(required_columns.keys())
    
    for col_name, format_verification_func in required_columns.items():
        assert format_verification_func(data[col_name]), f"Column {col_name} failed test {format_verification_func}"

def test_neighborhood_names(data: pd.DataFrame):
    
    known_cats = ['Queens', 'Manhattan', 'Brooklyn', 'Bronx', 'Staten Island']
    data_cats = data['neighbourhood_group'].unique()
    
    assert set(known_cats) == set(data_cats)

def test_room_type(data: pd.DataFrame):
    
    known_cats = ['Private room', 'Entire home/apt', 'Shared room']
    data_cats = data['room_type'].unique()
    
    assert set(known_cats) == set(data_cats)

def test_proper_city_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    
    assert np.sum(~idx) == 0

def test_similar_neighborhood_distribution(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the 
    new data is significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()
    
    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold

def test_row_count(data: pd.DataFrame):
    
    assert 15000 < data.shape[0] < 100000

def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    
    assert data['price'].between(min_price, max_price).all()
    