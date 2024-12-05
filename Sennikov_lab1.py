import time
import numpy as np # type: ignore
from numba import njit, prange # type: ignore
from scipy.spatial import cKDTree # type: ignore
from functools import wraps

def measure(func):

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time.time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time.time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it


@measure
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:

    if len(timestamps1) == 0 or len(timestamps2) == 0:
        return []

    matching = [-1] * len(timestamps1)
    j = 0

    for i in range(len(timestamps1)):
        while (j + 1 < len(timestamps2) and abs(timestamps2[j + 1] - timestamps1[i]) < abs(timestamps2[j] - timestamps1[i])):
            j += 1

        matching[i] = j

    return matching


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def main():
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    matching = match_timestamps(timestamps1, timestamps2)


if __name__ == '__main__':
    main()