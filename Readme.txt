# Uber Fare Prediction — Model Deployment with Flask

A web app that predicts the price of an Uber trip in New York City.
The user fills in a form (pickup, dropoff, date and time, passengers, car, traffic, weather),
and the app returns the predicted fare using the model trained in Task 2.

The app looks and works like a ride app: a map, a pickup/dropoff form, ride types,
busy-time (surge) prices, promo codes, a "best time to go" chart, and a ride simulation.
There is also a phone version at `/mobile`.

---

## Table of Contents

1. [Project Files](#project-files)
2. [How to Run](#how-to-run)
3. [Using the App](#using-the-app)
4. [Part A — Model Comparison](#part-a--model-comparison)
5. [Part B — Saving the Model](#part-b--saving-the-model)
6. [Part C — Flask Application](#part-c--flask-application)
7. [Part D — Testing Evidence](#part-d--testing-evidence)
8. [Limitations and Next Steps](#limitations-and-next-steps)

---

## Project Files

```
Task3_Uber_Fare/
├── notebook.ipynb            # Task 2 pipeline + model comparison + saving + Stage 12 check
├── app.py                    # Flask app (loads the model once, checks inputs, predicts)
├── fare_pipeline.py          # Feature engineering + preprocessing used in training and in the app
├── final_pipeline.joblib     # Saved model together with its preprocessing
├── fare_range.json           # Price range shown under the fare (from test errors)
├── pickup_points.json        # Sample pickup points for the busy-areas heat map
├── test_app.py               # 25 automated tests
├── speed_benchmark.py        # Measures prediction speed
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html            # Web page (map + form + result)
│   └── mobile.html           # Phone page
└── screenshots/
    ├── 01_times_square_to_wall_street.png
    ├── 02_jfk_to_midtown.png
    ├── 03_empire_state_to_grand_central.png
    └── 04_invalid_input.png
```

Files created while the app runs (not needed for submission):
`predictions.log` (one line per prediction) and `activity.json` (finished rides).

---

## How to Run

**1. Install the libraries** (Python 3.12):

```bash
py -3.12 -m pip install -r requirements.txt
```

`requirements.txt`:

```
flask
pandas
numpy
scikit-learn
joblib
```

> Use the same scikit-learn version that was used to save `final_pipeline.joblib`.

**2. Start the app** from the project folder:

```bash
py -3.12 app.py
```

Wait until the model loads and you see:

```
Web page   : http://127.0.0.1:5000
Phone page : http://127.0.0.1:5000/mobile
 * Running on http://127.0.0.1:5000
```

**3. Open** http://127.0.0.1:5000 in the browser.

To stop the app press **Ctrl + C** in the terminal (only after you finish, not while the model is loading).

**4. Run the tests** (optional):

```bash
py -3.12 -m unittest test_app.py -v
```

---

## Using the App

1. Click **Ride** (or "Where to?").
2. Choose pickup and dropoff: search a place, click on the map, use the City / ✈ Airport
   switch, or type the coordinates.
3. Fill in the trip details: date and time, passengers, car condition, traffic, weather.
4. Click **Get fare estimate**.
5. The app shows the price for each ride type (UberX, Comfort, UberXL), the price breakdown,
   and a chart of the price at every hour of that day.
6. Optional: add a promo code, then click **Choose UberX** to watch a simulated ride.

If any input is wrong, no price is shown. A red box lists what needs fixing.

---

## Part A — Model Comparison

All models use the same Task 2 preprocessing and the same train/test split.
Scores are on the **test set**.

| Model | MAE ($) | RMSE ($) | R² | Training time |
|-------|---------|----------|-----|---------------|
| Linear Regression | 2.13 | 4.83 | 0.743 | ~3 s |
| Random Forest | 1.59 | 3.95 | 0.828 | ~212 s |
| **Gradient Boosting (tuned)** | **1.59** | **3.93** | **0.829** | ~28 s |

Cross-validation RMSE for Gradient Boosting: **4.00** (close to the test RMSE, so the model is stable).

Best Gradient Boosting settings: learning rate 0.05, 300 iterations, 31 leaves, 10 rows per leaf.

### Why Gradient Boosting was chosen

Gradient Boosting has the lowest RMSE (3.93) and the highest R² (0.829), so it makes the
smallest errors on unseen trips. Random Forest is almost the same on the metrics, so the choice
was made on speed: Gradient Boosting trains about 7 times faster (28 s vs 212 s), predicts one
trip in about [__] ms, and its saved file is much smaller, which suits a web app that answers
every request live. Linear Regression is the fastest and easiest to explain, but it is clearly
less accurate (MAE $2.13 vs $1.59) because it cannot learn non-linear patterns such as the JFK
flat fare. For interpretability, Gradient Boosting still gives feature importances, which show
that distance and airport trips drive the price, matching the EDA.

### Data leakage check

- The split into train and test was done **before** fitting any scaler or encoder.
- All preprocessing lives inside the pipeline, so it is fitted on training data only.
- No feature uses the fare itself.

---

## Part B — Saving the Model

The whole pipeline (feature engineering + preprocessing + model) is saved as **one file**:

```python
joblib.dump(final_pipeline, "final_pipeline.joblib")
```

Saving the pipeline (not only the model) means the app cannot forget a preprocessing step
or apply it in a different way.

### Relation to the Task 2 pipeline

The app uses the **same pipeline** as Task 2 with no changes:

1. **Feature engineering** (`TripFeatures` in `fare_pipeline.py`):
   - straight-line distance (haversine) and its log
   - distance to JFK, LaGuardia, Newark and Midtown
   - direction of the trip (bearing)
   - year, month, weekday, weekend flag, hour as sine/cosine
   - after Sept 2012 flag (the NYC tariff changed then) and distance × that flag
   - JFK ↔ Manhattan flag
2. **Preprocessing** (`ColumnTransformer`):
   - numbers → `RobustScaler`
   - Car Condition and Traffic Condition → `OrdinalEncoder` (ordered levels)
   - Weather → `OneHotEncoder`
3. **Model** → Gradient Boosting.

The app sends the **raw** inputs in the same column order as the training data
(`Car Condition`, `Weather`, `Traffic Condition`, `pickup_datetime`, coordinates,
`passenger_count`), and the pipeline does the rest.

---

## Part C — Flask Application

### How a request flows

```
Form → read_form() checks every field → make_trips() builds a DataFrame
     → model.predict() → business rules (ride type, surge, promo) → result page
```

- The model is loaded **once** when the app starts, not on every request.
- Every prediction is written to `predictions.log`.

### Routes

| Route | Method | What it does |
|-------|--------|--------------|
| `/` | GET | Shows the page with the form |
| `/` | POST | Checks the form, predicts, shows the result or the errors |
| `/predict` | POST | Same as above but returns JSON (used by the page without reloading); includes `latency_ms` |
| `/hours` | POST | Price for every hour of the chosen day (for the "Best time to go" chart) |
| `/activity` | GET / POST | Reads / saves finished rides |
| `/heat` | GET | Sample pickup points for the busy-areas map |
| `/mobile` | GET | Phone version of the app |

### Input checks

All checks run on the **server**, so they work even if the browser checks are skipped.
The limits are the same as the cleaning stage in Task 2.

| Field | Rule | Error message |
|-------|------|---------------|
| Coordinates | Required, a number | "Pickup latitude is required." / "… must be a number." |
| Latitude | 40.5 to 41.0 | "Pickup latitude must be between 40.5 and 41.0 (inside New York)." |
| Longitude | -74.3 to -73.7 | "Pickup longitude must be between -74.3 and -73.7 (inside New York)." |
| Pickup = dropoff | Not allowed | "Pickup and dropoff are the same place. Please choose two different places." |
| Passengers | Whole number 1–6 | "Number of passengers must be a whole number from 1 to 6." |
| Date and time | Jan 2009 – Jun 2015 (the range of the data) | "Pickup date must be between January 2009 and June 2015 …" |
| Car / traffic / weather | One of the known options | "Please choose a valid car condition." |
| 5–6 passengers | Need a bigger car | "Trips with 5 or 6 passengers need an UberXL." |

If the model itself fails, the user sees:
"Something went wrong while predicting. Please check your inputs." and the app keeps running.

### What comes from the model and what comes from business rules

| Part of the price | Source |
|-------------------|--------|
| **Trip fare** | The model prediction (always shown as is) |
| Ride type | UberX ×1.0, Comfort ×1.25, UberXL ×1.5 |
| Surge | Mon–Fri 16:00–20:00 ×1.25 "Rush hour"; Fri/Sat 22:00–24:00 and Sat/Sun 00:00–03:00 ×1.2 "Weekend night" |
| Promo codes | WELCOME10 (10% off), AIRPORT5 ($5 off airport trips), NIGHT20 (20% off 00:00–05:00) |
| Price range | Final price × 0.88 to × 1.14 (from the test-set errors) |

So the model decides the base price, and the app adds clear rules on top, the same way
a real ride app adds ride types and surge to a base fare.

---

## Part D — Testing Evidence

### Screenshots

Each screenshot shows the filled-in form and the result it produced.
All screenshots are in the `screenshots/` folder.

| # | File | Pickup → Dropoff | Date and time | Passengers / Ride | Trip fare (model) | Final price | Distance |
|---|------|------------------|---------------|-------------------|-------------------|-------------|----------|
| 1 | `01_times_square_to_wall_street.png` | Times Square (40.7580, -73.9855) → Wall Street (40.7069, -74.0113) | Wed 2014-03-05, 14:00 | 5 / UberXL (×1.5) | $19.62 | $29.43 | 6.08 km |
| 2 | `02_jfk_to_midtown.png` | JFK Airport → Midtown (40.7580, -73.9855) | Mon 2014-05-05, 14:00 | 5 / UberXL (×1.5) | $57.77 | $86.66 | 21.77 km |
| 3 | `03_empire_state_to_grand_central.png` | Empire State (40.7484, -73.9857) → Grand Central (40.7527, -73.9772) | Sat 2014-03-08, 09:00 | 1 / UberX | $__ | $__ | __ km |
| 4 | `04_invalid_input.png` | Pickup latitude = 30.7489 (outside New York) | Sat 2014-03-08, 09:00 | — | — | No price | — |

**Why these trips:**

- **Trip 1:** a normal city ride (about 6 km).
- **Trip 2:** an airport ride. The model learned the JFK flat fare from the data.
- **Trip 3:** a very short ride, to check that small trips get small prices.
- **Trip 4:** wrong input. The app does not predict and shows a red message:
  *"Pickup latitude must be between 40.5 and 41.0 (inside New York)."*

None of these trips fall in a surge period, so the final price is only
trip fare × ride type.

#### 1. Times Square → Wall Street
![Trip 1](screenshots/01_times_square_to_wall_street.png)

#### 2. JFK → Midtown
![Trip 2](screenshots/02_jfk_to_midtown.png)

#### 3. Empire State → Grand Central
![Trip 3](screenshots/03_empire_state_to_grand_central.png)

#### 4. Invalid input
![Invalid input](screenshots/04_invalid_input.png)

### Automated tests

`test_app.py` has **25 tests** using the Flask test client:

- **Form checks (8):** empty fields, letters instead of numbers, points outside New York,
  same pickup and dropoff, wrong passenger count, dates outside the data range.
- **Busy times (4):** Wed 17:00 → ×1.25, Sun 17:00 → ×1.0, Sat 01:00 → ×1.2, Wed 14:00 → ×1.0.
- **Prices:** promo codes, surge added to the price, ride types, activity saving,
  the web and phone pages open, and `/predict` returns `latency_ms`.

```bash
py -3.12 -m unittest test_app.py -v
```

Result: `Ran 25 tests ... OK`

### Example request and response (notebook Stage 12)

The notebook sends one valid and one invalid request to the app with `app.test_client()`.

**Valid request:** Times Square → Wall Street, Wed 2014-03-05 14:00, 1 passenger, UberX

```
POST /predict
→ model_fare: $__   final_fare: $__   distance_km: __
```

**Invalid request:** pickup and dropoff at the same point

```
POST /predict
→ errors: ["Pickup and dropoff are the same place. Please choose two different places."]
```

### Prediction speed

Measured with `speed_benchmark.py` and in notebook Stage 12 (200 calls):

| Measure | Model only | Full `/predict` request |
|---------|-----------|-------------------------|
| Median | __ ms | __ ms |
| 95% of calls under | __ ms | __ ms |

### Sanity check against the real NYC taxi tariff (Sept 2012)

Tariff: $2.50 start, $2.50 per mile, $0.50 per slow minute, JFK ↔ Manhattan $52 flat.

| Trip | Model | Tariff estimate | Verdict |
|------|-------|-----------------|---------|
| Midtown → Empire State, ~1 km | $7.41 | $5–8 | Fits |
| Midtown → Wall Street, ~6 km | $20.74 | $17–22 | Fits |
| JFK → Midtown | $58.52 | $52 flat + tolls | Fits |

The model was never given these rules. It learned the airport flat fare and the
price per distance from the trips themselves.

---

## Limitations and Next Steps

**Limitations**

- Prices are 2009–2015 prices, so later dates are refused.
- Car condition, traffic and weather barely change the fare (almost no correlation in the EDA).
- The model uses straight-line distance, not the road route (the route on the map is only for display).
- The app runs on Flask's development server (debug mode) on a local machine.

**Next steps**

- Retrain on recent trips.
- Add road distance and travel time as features.
- Serve with a production server (e.g. gunicorn) with debug off, on a cloud host.
- Watch `predictions.log` for changes in predicted fares over time.

---

