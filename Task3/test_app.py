import json
import logging
import os
import time
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from fare_pipeline import CAR_ORDER, EWR, JFK, LGA, TRAFFIC_ORDER, haversine_km

# the folder where app.py lives, so files are found from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# use the templates folder if index.html is there, otherwise the app folder
if os.path.exists(os.path.join(BASE_DIR, "templates", "index.html")):
    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
else:
    TEMPLATE_DIR = BASE_DIR

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# load the saved pipeline once, when the app starts
model = joblib.load(os.path.join(BASE_DIR, "final_pipeline.joblib"))

# likely price range, made in the notebook from the model's errors on the test set
range_file = os.path.join(BASE_DIR, "fare_range.json")
if os.path.exists(range_file):
    with open(range_file) as f:
        FARE_RANGE = json.load(f)
else:
    FARE_RANGE = {"low": 0.9, "high": 1.1}

# pickup points from the train set, for the busy-areas layer on the map
points_file = os.path.join(BASE_DIR, "pickup_points.json")
if os.path.exists(points_file):
    with open(points_file) as f:
        PICKUP_POINTS = json.load(f)
else:
    PICKUP_POINTS = []

# trips the user requested, shown on the Activity page
ACTIVITY_FILE = os.path.join(BASE_DIR, "activity.json")

# results of the final model on the test set (from the notebook), shown on the Account page
MODEL_INFO = {"name": "Tuned Gradient Boosting", "mae": 1.59, "rmse": 3.93, "r2": 0.829,
              "period": "January 2009 – June 2015"}

# every prediction is written to predictions.log (Flask messages still show in the terminal)
logger = logging.getLogger("predictions")
logger.setLevel(logging.INFO)
log_file = logging.FileHandler(os.path.join(BASE_DIR, "predictions.log"))
log_file.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(log_file)

# weather labels the model saw during training
WEATHER_OPTIONS = list(model.named_steps["prep"].named_transformers_["weather"].categories_[0])

# same limits as the cleaning stage in the notebook
LAT_RANGE = (40.5, 41.0)
LON_RANGE = (-74.3, -73.7)
FIRST_DATE = datetime(2009, 1, 1)
LAST_DATE = datetime(2015, 6, 30, 23, 59)

# same column order as the training data
COLUMNS = ["Car Condition", "Weather", "Traffic Condition", "pickup_datetime",
           "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
           "passenger_count"]

# ride types: price = model fare x multiplier (a business rule added after the model)
RIDE_TYPES = {"UberX": 1.0, "Comfort": 1.25, "UberXL": 1.5}

# promo codes, checked on the server
PROMO_CODES = {
    "WELCOME10": "10% off any trip",
    "AIRPORT5": "$5 off trips to or from an airport",
    "NIGHT20": "20% off trips between midnight and 5 AM",
}

# values shown when the page opens
DEFAULTS = {
    "pickup_latitude": "40.7580", "pickup_longitude": "-73.9855",
    "dropoff_latitude": "40.7069", "dropoff_longitude": "-74.0113",
    "pickup_datetime": "2014-03-05T14:00", "passenger_count": "1",
    "car_condition": "Good", "traffic_condition": "Flow Traffic",
    "weather": WEATHER_OPTIONS[0], "ride_type": "UberX", "promo_code": "", "service": "ride",
}


# check every field and return the clean values and a list of error messages
def read_form(form):
    values = {}
    errors = []

    # coordinates must be numbers inside New York
    coordinates = [
        ("pickup_latitude", "Pickup latitude", LAT_RANGE),
        ("pickup_longitude", "Pickup longitude", LON_RANGE),
        ("dropoff_latitude", "Dropoff latitude", LAT_RANGE),
        ("dropoff_longitude", "Dropoff longitude", LON_RANGE),
    ]
    for field, label, (low, high) in coordinates:
        text = str(form.get(field, "")).strip()
        if text == "":
            errors.append(f"{label} is required.")
            continue
        try:
            number = float(text)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not low <= number <= high:
            errors.append(f"{label} must be between {low} and {high} (inside New York).")
            continue
        values[field] = number

    # pickup and dropoff cannot be the same point
    if len(values) == 4 and (values["pickup_latitude"], values["pickup_longitude"]) == \
            (values["dropoff_latitude"], values["dropoff_longitude"]):
        errors.append("Pickup and dropoff are the same place. Please choose two different places.")

    # passengers must be a whole number from 1 to 6
    text = str(form.get("passenger_count", "")).strip()
    if text == "":
        errors.append("Number of passengers is required.")
    elif not text.isdigit() or not 1 <= int(text) <= 6:
        errors.append("Number of passengers must be a whole number from 1 to 6.")
    else:
        values["passenger_count"] = int(text)

    # the date must be readable and inside the dates the model learned from
    text = str(form.get("pickup_datetime", "")).strip()
    if text == "":
        errors.append("Pickup date and time is required.")
    else:
        try:
            pickup_time = datetime.fromisoformat(text)
            if FIRST_DATE <= pickup_time <= LAST_DATE:
                values["pickup_datetime"] = pickup_time
            else:
                errors.append("Pickup date must be between January 2009 and June 2015, "
                              "the dates the model was trained on.")
        except ValueError:
            errors.append("Pickup date and time could not be read.")

    # the text choices must be one of the known labels
    choices = [
        ("car_condition", "Car condition", CAR_ORDER),
        ("traffic_condition", "Traffic condition", TRAFFIC_ORDER),
        ("weather", "Weather", WEATHER_OPTIONS),
        ("ride_type", "Ride type", list(RIDE_TYPES)),
    ]
    for field, label, options in choices:
        if form.get(field) not in options:
            errors.append(f"Please choose a valid {label.lower()}.")
        else:
            values[field] = form.get(field)

    # 5 or 6 passengers only fit in an UberXL
    if values.get("passenger_count", 0) >= 5 and values.get("ride_type") in ("UberX", "Comfort"):
        errors.append("Trips with 5 or 6 passengers need an UberXL.")

    # the promo code is optional; it is checked later so a wrong code does not block the price
    values["promo_code"] = str(form.get("promo_code", "")).strip().upper()

    return values, errors


# one or more raw trips in the same format as the training data
def make_trips(values, hours=None):
    times = [values["pickup_datetime"]] if hours is None else \
        [values["pickup_datetime"].replace(hour=h, minute=0) for h in hours]
    rows = [{
        "Car Condition": values["car_condition"],
        "Weather": values["weather"],
        "Traffic Condition": values["traffic_condition"],
        "pickup_datetime": t,
        "pickup_longitude": values["pickup_longitude"],
        "pickup_latitude": values["pickup_latitude"],
        "dropoff_longitude": values["dropoff_longitude"],
        "dropoff_latitude": values["dropoff_latitude"],
        "passenger_count": values["passenger_count"],
    } for t in times]
    return pd.DataFrame(rows)[COLUMNS]


# true if the pickup or the dropoff is within 2 km of an airport
def near_airport(values):
    for lat, lon in (JFK, LGA, EWR):
        if haversine_km(values["pickup_latitude"], values["pickup_longitude"], lat, lon) < 2 or \
                haversine_km(values["dropoff_latitude"], values["dropoff_longitude"], lat, lon) < 2:
            return True
    return False


# busy times, like Uber's surge pricing (a business rule on top of the model)
def surge_for(pickup_time):
    day, hour = pickup_time.weekday(), pickup_time.hour
    # Monday to Friday, 4 PM to 8 PM
    if day < 5 and 16 <= hour < 20:
        return 1.25, "Rush hour"
    # Friday and Saturday nights, 10 PM to 3 AM
    if (day in (4, 5) and hour >= 22) or (day in (5, 6) and hour < 3):
        return 1.2, "Weekend night"
    return 1.0, ""


# the discount and a message for a promo code
def apply_promo(code, fare, values):
    if code == "":
        return 0.0, ""
    if code not in PROMO_CODES:
        return 0.0, f"Promo code {code} was not found."
    if code == "WELCOME10":
        return round(fare * 0.10, 2), "WELCOME10 applied: 10% off."
    if code == "AIRPORT5":
        if near_airport(values):
            return round(min(5.0, fare), 2), "AIRPORT5 applied: $5 off this airport trip."
        return 0.0, "AIRPORT5 only works for trips to or from an airport."
    if code == "NIGHT20":
        if values["pickup_datetime"].hour < 5:
            return round(fare * 0.20, 2), "NIGHT20 applied: 20% off this night trip."
        return 0.0, "NIGHT20 only works between midnight and 5 AM."
    return 0.0, ""


# the full price: model fare, ride type, busy time, promo, final fare and likely range
def price_details(values):
    model_fare = round(float(model.predict(make_trips(values))[0]), 2)
    multiplier = RIDE_TYPES[values["ride_type"]]
    ride_fare = round(model_fare * multiplier, 2)
    surge, surge_label = surge_for(values["pickup_datetime"])
    surge_fare = round(ride_fare * surge, 2)
    discount, promo_message = apply_promo(values["promo_code"], surge_fare, values)
    final_fare = round(surge_fare - discount, 2)
    distance = haversine_km(values["pickup_latitude"], values["pickup_longitude"],
                            values["dropoff_latitude"], values["dropoff_longitude"])
    details = {
        "model_fare": model_fare,
        "ride_type": values["ride_type"],
        "multiplier": multiplier,
        "ride_fare": ride_fare,
        "surge": surge,
        "surge_label": surge_label,
        "surge_amount": round(surge_fare - ride_fare, 2),
        "surge_fare": surge_fare,
        "discount": discount,
        "promo_message": promo_message,
        "final_fare": final_fare,
        "low": round(final_fare * FARE_RANGE["low"], 2),
        "high": round(final_fare * FARE_RANGE["high"], 2),
        "distance_km": round(float(distance), 2),
        "airport": near_airport(values),
        "rides": {name: round(model_fare * m * surge, 2) for name, m in RIDE_TYPES.items()},
    }
    logger.info("pickup=(%.4f, %.4f) dropoff=(%.4f, %.4f) time=%s passengers=%d ride=%s surge=%.2f promo=%s "
                "model_fare=%.2f final_fare=%.2f",
                values["pickup_latitude"], values["pickup_longitude"],
                values["dropoff_latitude"], values["dropoff_longitude"],
                values["pickup_datetime"], values["passenger_count"], values["ride_type"], surge,
                values["promo_code"] or "-", model_fare, final_fare)
    return details


# the page: shows the form, and the price after the form is sent
@app.route("/", methods=["GET", "POST"])
def index():
    form = dict(DEFAULTS)
    details = None
    errors = []

    if request.method == "POST":
        # keep what the user typed so the form is not cleared
        form = request.form.to_dict()
        values, errors = read_form(form)
        if not errors:
            try:
                details = price_details(values)
            except Exception:
                errors.append("Something went wrong while predicting. Please check your inputs.")

    return render_template("index.html", form=form, details=details, errors=errors,
                           car_options=CAR_ORDER, traffic_options=TRAFFIC_ORDER,
                           weather_options=WEATHER_OPTIONS, ride_types=RIDE_TYPES,
                           promo_codes=PROMO_CODES, model_info=MODEL_INFO)


# live price: the page calls this in the background every time the trip changes
@app.route("/predict", methods=["POST"])
def predict_live():
    values, errors = read_form(request.form)
    if errors:
        return jsonify({"errors": errors})
    try:
        # time the prediction, so the app can show how fast it is
        start = time.perf_counter()
        details = price_details(values)
        details["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return jsonify(details)
    except Exception:
        return jsonify({"errors": ["Something went wrong while predicting. Please check your inputs."]})


# the phone version of the app: same model, same checks, same /predict
@app.route("/mobile")
def mobile():
    return render_template("mobile.html", form=DEFAULTS, car_options=CAR_ORDER,
                           traffic_options=TRAFFIC_ORDER, weather_options=WEATHER_OPTIONS,
                           promo_codes=PROMO_CODES)


# the same trip at every hour of the day, to find the cheapest and the most expensive time
@app.route("/hours", methods=["POST"])
def fares_by_hour():
    values, errors = read_form(request.form)
    if errors:
        return jsonify({"errors": errors})
    fares = model.predict(make_trips(values, hours=range(24))) * RIDE_TYPES[values["ride_type"]]
    hours = []
    for h, f in enumerate(fares):
        surge, label = surge_for(values["pickup_datetime"].replace(hour=h))
        hours.append({"hour": h, "fare": round(float(f), 2), "surge": surge, "label": label,
                      "total": round(float(f) * surge, 2)})
    cheapest = min(hours, key=lambda x: x["total"])
    priciest = max(hours, key=lambda x: x["total"])
    return jsonify({"hours": hours, "cheapest": cheapest, "priciest": priciest})


# the Activity page: read the saved trips, or save a new one (keeps the last 20)
@app.route("/activity", methods=["GET", "POST"])
def activity():
    trips = []
    if os.path.exists(ACTIVITY_FILE):
        with open(ACTIVITY_FILE) as f:
            trips = json.load(f)
    if request.method == "POST":
        trip = request.get_json(silent=True) or {}
        # keep only the fields the page needs
        saved = {key: trip.get(key) for key in
                 ["service", "label", "pickup", "dropoff", "pickup_name", "dropoff_name", "time", "price"]}
        saved["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        trips = [saved] + trips[:19]
        with open(ACTIVITY_FILE, "w") as f:
            json.dump(trips, f)
        return jsonify({"saved": True})
    return jsonify(trips)


# pickup points from the train set, for the busy-areas layer
@app.route("/heat")
def heat_points():
    return jsonify(PICKUP_POINTS)


if __name__ == "__main__":
    app.run(debug=True)