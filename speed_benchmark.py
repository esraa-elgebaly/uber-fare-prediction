import statistics
import time

from app import DEFAULTS, app, make_trips, model, read_form

# the same trip the app opens with (Times Square -> Wall Street), cleaned by the app's own checks
values, errors = read_form(DEFAULTS)
trip = make_trips(values)
client = app.test_client()


# run a function many times and print how long one call takes
def measure(name, run, calls=200):
    run()  # first call is slower, so it is not counted
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    print(f"{name}")
    print(f"  median : {statistics.median(times):6.2f} ms")
    print(f"  p95    : {times[int(0.95 * calls)]:6.2f} ms")
    print(f"  max    : {times[-1]:6.2f} ms\n")


# 1) the model alone: raw trip -> saved pipeline -> fare
measure("Model only (pipeline.predict)", lambda: model.predict(trip))

# 2) the whole request the page sends: checks + pipeline + pricing rules + JSON
measure("Full request (POST /predict)", lambda: client.post("/predict", data=DEFAULTS))