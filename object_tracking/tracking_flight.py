import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from drone_control.tello import Tello
from .tracker import annotate, detect_board


VIDEO_URL = "udp://0.0.0.0:11111"
WINDOW = "Tello checkerboard tracking"


def tracking_options():
    parser = argparse.ArgumentParser(description="Fly the planned route and track during hover.")
    parser.add_argument("--hover-seconds", type=float, default=10,
                        help="tracking duration after the first live frame (default: 10)")
    parser.add_argument("--headless", action="store_true", help="print results without a preview window")
    args = parser.parse_args()
    if not math.isfinite(args.hover_seconds) or args.hover_seconds <= 0:
        parser.error("--hover-seconds must be finite and positive")
    return args


def require_ok(tello, command, timeout=None):
    reply = tello.send_command(command, timeout=timeout)
    if reply is None or reply.lower() != "ok":
        raise RuntimeError(f"{command}: {reply if reply is not None else 'no confirmed reply'}")


def track_stream(tello, duration, headless, output):
    writer = csv.writer(output)
    writer.writerow(("elapsed_s", "x_px", "y_px", "status"))
    output.flush()
    capture = cv2.VideoCapture(
        VIDEO_URL, cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000],
    )
    window_open = False
    try:
        if not capture.isOpened():
            raise RuntimeError("could not open the Tello video stream")
        started = None
        keepalive_at = time.monotonic()
        while started is None or time.monotonic() - started < duration:
            if time.monotonic() >= keepalive_at:
                require_ok(tello, "command", timeout=2)
                keepalive_at = time.monotonic() + 3
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok:
                raise RuntimeError("Tello video stopped delivering frames")
            if started is None:
                started = now
            elapsed = now - started
            if elapsed >= duration:
                break
            detection = detect_board(frame)
            if detection is None:
                writer.writerow((f"{elapsed:.3f}", "", "", "target lost"))
                message = "target lost"
            else:
                x, y = detection[1]
                writer.writerow((f"{elapsed:.3f}", x, y, "detected"))
                message = f"x={x} px y={y} px"
            output.flush()
            if headless:
                print(f"{elapsed:.3f}s {message}", flush=True)
            else:
                cv2.imshow(WINDOW, annotate(frame, detection))
                window_open = True
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        if window_open:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                # the window may already have been closed
                pass


def fly_and_track(commands, duration, headless):
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    tracking_path = log_dir / f"{timestamp}_tracking.csv"
    command_path = log_dir / f"{timestamp}.txt"
    tello = Tello()
    takeoff_attempted = False
    stream_attempted = False
    cleanup_failed = False
    try:
        require_ok(tello, "command", timeout=2)
        takeoff_attempted = True
        require_ok(tello, "takeoff")
        for command in commands:
            require_ok(tello, command)
        stream_attempted = True
        require_ok(tello, "streamon", timeout=2)
        with tracking_path.open("w", newline="", encoding="utf-8") as output:
            print(f"Tracking log: {tracking_path}")
            print("Destination reached. Tracking during stationary hover.")
            track_stream(tello, duration, headless, output)
    finally:
        try:
            for needed, command in ((takeoff_attempted, "land"), (stream_attempted, "streamoff")):
                if needed:
                    try:
                        require_ok(tello, command)
                    except (RuntimeError, OSError) as exc:
                        cleanup_failed = True
                        print(f"Cleanup command failed: {exc}", file=sys.stderr)
        finally:
            tello.on_close()
            command_path.write_text("".join(stat.return_stats() for stat in tello.get_log()),
                                    encoding="utf-8")
            print(f"Command log: {command_path}")
    if cleanup_failed:
        raise RuntimeError("Tello cleanup did not complete")
