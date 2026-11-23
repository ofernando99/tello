import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

from drone_control.tello import Tello
from .tracking_flight import require_ok, track_stream


def main():
    parser = argparse.ArgumentParser(description="Track a checkerboard while the Tello stays on the ground.")
    parser.add_argument("--seconds", type=float, default=60,
                        help="tracking duration after the first frame (default: 60)")
    parser.add_argument("--headless", action="store_true", help="print results without a preview window")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_path = log_dir / f"{timestamp}_ground_tracking.csv"
    tello = Tello()
    try:
        require_ok(tello, "command", timeout=2)
        try:
            require_ok(tello, "streamon", timeout=2)
            with log_path.open("w", newline="", encoding="utf-8") as output:
                print(f"Tracking log: {log_path}")
                track_stream(tello, args.seconds, args.headless, output)
        finally:
            try:
                require_ok(tello, "streamoff", timeout=2)
            except (RuntimeError, OSError) as exc:
                print(f"Could not confirm video stream shutdown: {exc}", file=sys.stderr)
    finally:
        tello.on_close()


if __name__ == "__main__":
    main()
