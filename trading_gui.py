"""One-shot, image-gated GUI automation. Live clicks require --execute."""

import argparse
import sys

from safe_executor import EmergencyStop, SafeExecutor
from screen_detector import ImageDetector, Region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a screen image and optionally click a configured coordinate."
    )
    parser.add_argument("--template", required=True, help="Path to the reference image.")
    parser.add_argument(
        "--region",
        required=True,
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen region to inspect.",
    )
    parser.add_argument(
        "--click",
        required=True,
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Coordinate to click if the image is detected.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Template-match confidence from 0 to 1 (default: 0.9).",
    )
    parser.add_argument(
        "--log-file",
        default="gui_actions.log",
        help="Append action records to this text file.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable the mouse click; without this flag, run detection only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    region = Region(*args.region)
    click_x, click_y = args.click
    detector = ImageDetector(args.template, region, args.threshold)

    try:
        with SafeExecutor(args.log_file) as executor:
            executor.check_kill_switch()
            detection = detector.find()
            executor.check_kill_switch()

            if detection is None:
                print("Template not found; no click performed.")
                return 0

            print(
                f"Template detected at {detection.center} "
                f"(confidence {detection.confidence:.3f})."
            )
            if args.execute:
                executor.move_and_click(click_x, click_y)
            else:
                executor.log_action("DRY_RUN", click_x, click_y)
                print("Detection-only mode; no click performed. Use --execute to enable clicks.")
    except EmergencyStop:
        print("ESC kill switch activated; automation stopped.", file=sys.stderr)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
