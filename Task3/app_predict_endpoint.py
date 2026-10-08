"""
Add-on for the Uber Task 3 Flask app: a JSON endpoint (/predict) that the
mobile-style UI (uber_mobile_app.html) calls.

It reuses the SAME saved pipeline and the SAME validation rules as the web
form, so every prediction goes through the Task 2 pipeline end-to-end.

Put these routes in your existing app.py (merge, don't duplicate), or run
this file on a second port (e.g. 5001) and set that URL in the mobile UI
settings (the gear button).
"""
import time
import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s")
log = logging.getLogger("predict")

# ---------------------------------------------------------------- load once
PIPELINE_PATH = "final_pipeline.joblib"          # same file as the form route
pipeline = joblib.load(PIPELINE_PATH)

# ------------------------------------------------------- validation rules
# identical to the README / Task 2 cleaning rules
LAT_RANGE  = (40.5, 41.0)
LON_RANGE  = (-74.3, -73.7)
YEAR_RANGE = (2009, 2015)          # months 1-6 for 2015, per training data
PAX_RANGE  = (1, 6)
CARS       = {"Bad", "Good", "Very Good", "Excellent"}
TRAFFIC    = {"Flow Traffic", "Dense Traffic", "Congested Traffic"}
WEATHER    = {"cloudy", "rainy", "stormy", "sunny", "windy"}


def validate(d: dict):
    """Return an error message or None. Never raises on bad user input."""
    if d is None:
        return "Request body must be JSON."
    for k in ("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
              "pickup_datetime", "passengers", "car_condition",
              "traffic", "weather"):
        if k not in d or d[k] in (None, ""):
            return f"Missing field: {k}"

    try:
        plat, plon = float(d["pickup_lat"]), float(d["pickup_lon"])
        dlat, dlon = float(d["dropoff_lat"]), float(d["dropoff_lon"])
        pax = int(d["passengers"])
    except (TypeError, ValueError):
        return "Coordinates and passengers must be numbers."

    if not (LAT_RANGE[0] <= plat <= LAT_RANGE[1] and
            LAT_RANGE[0] <= dlat <= LAT_RANGE[1]):
        return "Latitude must be between 40.5 and 41.0 (New York)."
    if not (LON_RANGE[0] <= plon <= LON_RANGE[1] and
            LON_RANGE[0] <= dlon <= LON_RANGE[1]):
        return "Longitude must be between -74.3 and -73.7 (New York)."
    if not (PAX_RANGE[0] <= pax <= PAX_RANGE[1]):
        return "Passengers must be between 1 and 6."

    dt = pd.to_datetime(d["pickup_datetime"], errors="coerce")
    if pd.isna(dt):
        return "Pickup date and time could not be read."
    if not (YEAR_RANGE[0] <= dt.year <= YEAR_RANGE[1]):
        return "Pickup date must be between January 2009 and June 2015."
    if dt.year == 2015 and dt.month > 6:
        return "Pickup date must be between January 2009 and June 2015."

    if d["car_condition"] not in CARS:
        return f"Car condition must be one of {sorted(CARS)}."
    if d["traffic"] not in TRAFFIC:
        return f"Traffic must be one of {sorted(TRAFFIC)}."
    if d["weather"] not in WEATHER:
        return f"Weather must be one of {sorted(WEATHER)}."

    # same pickup and dropoff -> distance 0, model would guess the base fare
    if abs(plat - dlat) < 1e-9 and abs(plon - dlon) < 1e-9:
        return "Pickup and dropoff are the same point."
    return None


# ---------------------------------------------------- raw row for pipeline
# IMPORTANT: these keys must match the RAW column names your TripFeatures
# transformer expects (the same columns the training DataFrame had before
# feature engineering). Check fare_pipeline.py and rename here if needed.
def to_raw_row(d: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "pickup_latitude":    float(d["pickup_lat"]),
        "pickup_longitude":   float(d["pickup_lon"]),
        "dropoff_latitude":   float(d["dropoff_lat"]),
        "dropoff_longitude":  float(d["dropoff_lon"]),
        "pickup_datetime":    pd.to_datetime(d["pickup_datetime"]),
        "passenger_count":    int(d["passengers"]),
        "car_condition":      d["car_condition"],
        "traffic":            d["traffic"],
        "weather":            d["weather"],
    }])


# --------------------------------------------------------------- the route
def register(app: Flask):

    @app.route("/predict", methods=["POST"])
    def predict():
        t0 = time.perf_counter()
        d = request.get_json(silent=True)

        err = validate(d)
        if err:
            return jsonify({"error": err}), 400

        try:
            row = to_raw_row(d)
            fare = float(pipeline.predict(row)[0])
        except Exception as exc:                      # never crash the app
            log.exception("prediction failed")
            return jsonify({"error":
                            f"Prediction failed: {exc}"}), 500

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info("fare=%.2f  latency=%.1f ms  pax=%s",
                 fare, latency_ms, d["passengers"])
        return jsonify({"fare": round(fare, 2),
                        "latency_ms": round(latency_ms, 1)})


if __name__ == "__main__":
    app = Flask(__name__)
    register(app)
    app.run(debug=True)          # 127.0.0.1:5000, same URL as the form app
