import argparse
import time
from pathlib import Path

import cv2
import numpy as np


SIZE = 180
EDGE_KERNEL = np.ones((3, 3), dtype=np.uint8)
# fifteen by ten inner corners means sixteen by eleven squares
COLS, ROWS = 16, 11
SQUARES = np.indices((ROWS, COLS)).sum(axis=0) % 2 == 0
DESTINATION = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
INSETS = np.arange(0, 0.101, 0.02)
X, Y = np.meshgrid(np.arange(COLS) + 0.5, np.arange(ROWS) + 0.5)
SAMPLE_X = (SIZE * (INSETS[:, None, None] +
                   (1 - 2 * INSETS[:, None, None]) * X / COLS)).astype(int)
SAMPLE_Y = (SIZE * (INSETS[:, None, None] +
                   (1 - 2 * INSETS[:, None, None]) * Y / ROWS)).astype(int)


def detect_board(frame):
    """Return the board quadrilateral and its pixel centre, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 100)
    detection = find_board(gray, edges)
    if detection is not None:
        return detection
    # close small gaps in the border before trying again
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, EDGE_KERNEL)
    return find_board(gray, closed)


def find_board(gray, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < gray.size * 0.005:
            break
        quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        corners = quad.reshape(4, 2).astype(np.float32)
        for rotation in (0, 1):
            source = np.roll(corners, rotation, axis=0)
            transform = cv2.getPerspectiveTransform(source, DESTINATION)
            flat = cv2.warpPerspective(gray, transform, (SIZE, SIZE))
            samples = flat[SAMPLE_Y[:, None], SAMPLE_X[None, :]]
            first = samples[..., SQUARES]
            second = samples[..., ~SQUARES]
            # check both checkerboard polarities
            contrast = np.maximum(
                np.percentile(first, 15, axis=-1) - np.percentile(second, 85, axis=-1),
                np.percentile(second, 15, axis=-1) - np.percentile(first, 85, axis=-1),
            )
            score = contrast.max()
            if score >= 25 and score > best_score:
                centre = cv2.perspectiveTransform(
                    np.float32([[[SIZE / 2, SIZE / 2]]]), np.linalg.inv(transform)
                )[0, 0]
                best = corners.astype(int), tuple(map(round, centre))
                best_score = score
    return best


def annotate(frame, detection):
    label = "target lost"
    if detection is not None:
        corners, centre = detection
        cv2.polylines(frame, [corners.astype(np.int32)], True, (80, 240, 100), 2)
        cv2.drawMarker(frame, centre, (80, 240, 100), cv2.MARKER_CROSS, 20, 2)
        label = f"x={centre[0]} px   y={centre[1]} px"
    cv2.rectangle(frame, (8, 8), (250, 44), (20, 20, 20), -1)
    cv2.putText(frame, label, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Track a 15 x 10 inner-corner checkerboard.")
    parser.add_argument("source", help="video path, camera index or video URL")
    parser.add_argument("--headless", action="store_true", help="print coordinates per frame")
    args = parser.parse_args()
    source = args.source
    if source.isdecimal() and not Path(source).exists():
        source = int(source)
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            parser.exit(1, f"could not open video source: {args.source}\n")
        delay = max(1, round(1000 / (capture.get(cv2.CAP_PROP_FPS) or 30)))
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            detection = detect_board(frame)
            if args.headless:
                print(detection[1] if detection is not None else "target lost", flush=True)
            else:
                cv2.imshow("Checkerboard tracker", annotate(frame, detection))
                elapsed = round(1000 * (time.perf_counter() - started))
                if (cv2.waitKey(max(1, delay - elapsed)) & 0xFF) in (27, ord("q")):
                    break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
