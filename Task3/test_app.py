import os
import tempfile
import unittest

import app as app_module
from datetime import datetime

from app import DEFAULTS, app, read_form, surge_for


# a valid trip to start from in every test
def trip(**changes):
    data = dict(DEFAULTS)
    data.update(changes)
    return data


class FormChecks(unittest.TestCase):
    # a normal trip has no errors
    def test_valid_trip(self):
        values, errors = read_form(trip())
        self.assertEqual(errors, [])

    # an empty field must give a message
    def test_missing_pickup(self):
        values, errors = read_form(trip(pickup_latitude=""))
        self.assertIn("Pickup latitude is required.", errors)

    # text instead of a number must give a message
    def test_not_a_number(self):
        values, errors = read_form(trip(dropoff_longitude="abc"))
        self.assertIn("Dropoff longitude must be a number.", errors)

    # a point outside New York must give a message
    def test_outside_new_york(self):
        values, errors = read_form(trip(pickup_latitude="30"))
        self.assertTrue(any("inside New York" in e for e in errors))

    # the same pickup and dropoff must give a message
    def test_same_place(self):
        values, errors = read_form(trip(dropoff_latitude="40.7580", dropoff_longitude="-73.9855"))
        self.assertTrue(any("same place" in e for e in errors))

    # a date after June 2015 must give a message
    def test_date_out_of_range(self):
        values, errors = read_form(trip(pickup_datetime="2020-01-01T10:00"))
        self.assertTrue(any("between January 2009 and June 2015" in e for e in errors))

    # 7 passengers is not allowed
    def test_too_many_passengers(self):
        values, errors = read_form(trip(passenger_count="7"))
        self.assertTrue(any("1 to 6" in e for e in errors))

    # 6 passengers need an UberXL
    def test_six_passengers_need_xl(self):
        values, errors = read_form(trip(passenger_count="6", ride_type="UberX"))
        self.assertIn("Trips with 5 or 6 passengers need an UberXL.", errors)


class BusyTimes(unittest.TestCase):
    # Wednesday 5 PM is rush hour
    def test_rush_hour(self):
        self.assertEqual(surge_for(datetime(2014, 3, 5, 17, 0))[0], 1.25)

    # Sunday 5 PM is not rush hour
    def test_sunday_afternoon(self):
        self.assertEqual(surge_for(datetime(2014, 3, 9, 17, 0))[0], 1.0)

    # Saturday 1 AM is a weekend night
    def test_weekend_night(self):
        self.assertEqual(surge_for(datetime(2014, 3, 8, 1, 0))[0], 1.2)

    # Wednesday 2 PM has no surge
    def test_normal_time(self):
        self.assertEqual(surge_for(datetime(2014, 3, 5, 14, 0))[0], 1.0)


class Prices(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    # a valid trip returns a fare and a range around it
    def test_predict_returns_fare(self):
        result = self.client.post("/predict", data=trip()).get_json()
        self.assertGreater(result["final_fare"], 0)
        self.assertLessEqual(result["low"], result["final_fare"])
        self.assertGreaterEqual(result["high"], result["final_fare"])

    # Comfort costs more than UberX for the same trip
    def test_ride_type_multiplier(self):
        x = self.client.post("/predict", data=trip(ride_type="UberX")).get_json()
        comfort = self.client.post("/predict", data=trip(ride_type="Comfort")).get_json()
        self.assertAlmostEqual(comfort["ride_fare"], round(x["model_fare"] * 1.25, 2), places=2)

    # WELCOME10 takes 10% off
    def test_welcome10(self):
        result = self.client.post("/predict", data=trip(promo_code="welcome10")).get_json()
        self.assertAlmostEqual(result["discount"], round(result["surge_fare"] * 0.10, 2), places=2)

    # AIRPORT5 does nothing for a city trip
    def test_airport5_city_trip(self):
        result = self.client.post("/predict", data=trip(promo_code="AIRPORT5")).get_json()
        self.assertEqual(result["discount"], 0)

    # AIRPORT5 takes $5 off a JFK trip
    def test_airport5_jfk_trip(self):
        result = self.client.post("/predict", data=trip(promo_code="AIRPORT5",
                                                        dropoff_latitude="40.6413",
                                                        dropoff_longitude="-73.7781")).get_json()
        self.assertEqual(result["discount"], 5.0)

    # a wrong code gives a message and no discount
    def test_unknown_code(self):
        result = self.client.post("/predict", data=trip(promo_code="FREE100")).get_json()
        self.assertEqual(result["discount"], 0)
        self.assertIn("not found", result["promo_message"])

    # at rush hour the price goes up by 25%
    def test_surge_in_price(self):
        result = self.client.post("/predict", data=trip(pickup_datetime="2014-03-05T17:00")).get_json()
        self.assertEqual(result["surge"], 1.25)
        self.assertAlmostEqual(result["final_fare"], round(result["ride_fare"] * 1.25, 2), places=2)

    # the hour chart has 24 prices
    def test_hours(self):
        result = self.client.post("/hours", data=trip()).get_json()
        self.assertEqual(len(result["hours"]), 24)

    # a bad form gives errors, not a crash
    def test_predict_with_errors(self):
        result = self.client.post("/predict", data=trip(pickup_latitude="")).get_json()
        self.assertIn("errors", result)

    # a requested trip is saved and shown on the Activity page (in a temporary file, not the real one)
    def test_activity(self):
        app_module.ACTIVITY_FILE = os.path.join(tempfile.mkdtemp(), "activity.json")
        self.client.post("/activity", json={"service": "ride", "label": "UberX", "price": 12.5,
                                            "pickup": [40.758, -73.9855], "dropoff": [40.7069, -74.0113]})
        trips = self.client.get("/activity").get_json()
        self.assertEqual(trips[0]["price"], 12.5)

    # the page opens
    def test_page_opens(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    # the phone version opens
    def test_mobile_page_opens(self):
        self.assertEqual(self.client.get("/mobile").status_code, 200)

    # the live price also says how long the prediction took
    def test_latency_returned(self):
        result = self.client.post("/predict", data=trip()).get_json()
        self.assertGreaterEqual(result["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()