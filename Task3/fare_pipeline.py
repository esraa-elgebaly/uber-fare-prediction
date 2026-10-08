import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# fixed places in degrees (lat, lon)
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
MIDTOWN = (40.7580, -73.9855)

# real order of the labels, from worst to best
CAR_ORDER = ["Bad", "Good", "Very Good", "Excellent"]
TRAFFIC_ORDER = ["Flow Traffic", "Dense Traffic", "Congested Traffic"]


# distance in km between two points given in degrees
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# builds the EDA features from the raw trip columns
class TripFeatures(BaseEstimator, TransformerMixin):
    # nothing to learn, only fixed formulas
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        plat, plon = out["pickup_latitude"], out["pickup_longitude"]
        dlat, dlon = out["dropoff_latitude"], out["dropoff_longitude"]

        # trip length, and its log because the fare curve bends on long trips
        out["distance"] = haversine_km(plat, plon, dlat, dlon)
        out["log_distance"] = np.log1p(out["distance"])

        # closest end of the trip to each airport and to Midtown
        for name, (la, lo) in {"jfk": JFK, "lga": LGA, "ewr": EWR, "midtown": MIDTOWN}.items():
            out[f"{name}_dist"] = np.minimum(haversine_km(plat, plon, la, lo),
                                             haversine_km(dlat, dlon, la, lo))

        # direction of the trip
        out["bearing"] = np.degrees(np.arctan2(dlon - plon, dlat - plat))

        # time parts from the pickup date
        dt = pd.to_datetime(out["pickup_datetime"])
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["weekday"] = dt.dt.dayofweek
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # hour as a circle, because hour 23 is next to hour 0
        out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

        # the price per km jumped in September 2012
        out["after_2012"] = (dt >= "2012-09-01").astype(int)
        out["dist_after_2012"] = out["distance"] * out["after_2012"]

        # JFK to Manhattan trips have a fixed fare
        pick_jfk = haversine_km(plat, plon, *JFK) < 2
        drop_jfk = haversine_km(dlat, dlon, *JFK) < 2
        pick_mid = haversine_km(plat, plon, *MIDTOWN) < 5
        drop_mid = haversine_km(dlat, dlon, *MIDTOWN) < 5
        out["is_jfk_manhattan"] = ((pick_jfk & drop_mid) | (drop_jfk & pick_mid)).astype(int)

        return out.drop(columns=["pickup_datetime"])


# numeric columns that go to the scaler
NUM_COLS = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
            "passenger_count", "distance", "log_distance",
            "jfk_dist", "lga_dist", "ewr_dist", "midtown_dist", "bearing",
            "year", "month", "weekday", "is_weekend", "hour_sin", "hour_cos",
            "after_2012", "dist_after_2012", "is_jfk_manhattan"]


# the full pipeline: features -> scaling and encoding -> model on log1p(fare)
def build_pipeline(model):
    prep = ColumnTransformer([
        ("num", RobustScaler(), NUM_COLS),
        ("ordered", OrdinalEncoder(categories=[CAR_ORDER, TRAFFIC_ORDER],
                                   handle_unknown="use_encoded_value", unknown_value=-1),
         ["Car Condition", "Traffic Condition"]),
        ("weather", OneHotEncoder(handle_unknown="ignore"), ["Weather"]),
    ])
    return Pipeline([
        ("features", TripFeatures()),
        ("prep", prep),
        ("model", TransformedTargetRegressor(regressor=model, func=np.log1p,
                                             inverse_func=np.expm1, check_inverse=False)),
    ])
