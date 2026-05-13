"""
Download and rotate the initial four-event source-array test set.

Edit BASE_CONFIG for your station and output path, then run:

    python scripts/source_array_download_test_events.py
"""

from source_array_download_rotate import CONFIG, main


BASE_CONFIG = dict(CONFIG)
NETWORK = "G"
STATION = "CRZF"

BASE_CONFIG.update(
    {
        "run_name": f"source_array_{NETWORK}_{STATION}_1994_1999_test_events",
        # Set this to the single receiver station you want to use.
        "network": NETWORK,
        "station": STATION,
        "preferred_location": "00",
        "channels": ["BH?"],
        # Put your longer event list in a text file and set this path.
        # If this is not None, it is used instead of the inline event_lines below.
        "event_list_file": None,
        "event_lines": [
            "1994JUL13 11:45:23-7.532 127.77",
            "1995DEC25 04:43:24 -6.903 129.151",
            "1997DEC22 02:05:50 -5.495 147.867",
            "1999APR05 11:08:04 -5.591 149.568",
        ],
        "min_depth_km": None,
        "max_depth_km": None,
        "tstart": 0.0,
        "tend": 1800.0,
        "phase_list_for_report": ["P"],
        "prompt_time_window": True,
        "downsample": True,
        "sampling_rate": 5.0,
        "download_noise": True,
        "rotate_after_download": True,
    }
)


if __name__ == "__main__":
    main(BASE_CONFIG)
