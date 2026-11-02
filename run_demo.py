#!/usr/bin/env python3
"""
Soccer Offside Line Detection & Vanishing Point Analysis
Multi-Case Runner Script

Author: Vasilis Stavropoulos
GitHub: https://github.com/VDitch
LinkedIn: https://www.linkedin.com/in/vasilis-stavropoulos/
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CASES = {
    "1": {
        "name": "Case 1: Argentina vs England (World Cup)",
        "script": BASE_DIR / "cases" / "01_argentina_vs_england" / "main.py",
        "description": "Historical broadcast calibration, mowing pattern analysis, and vanishing point line projection."
    },
    "2": {
        "name": "Case 2: Manchester City vs Manchester United (2022 Premier League)",
        "script": BASE_DIR / "cases" / "02_man_city_vs_man_united_2022" / "main.py",
        "description": "Controversial offside geometry, CLAHE-enhanced edge detection, and central-crop jersey classification."
    },
    "3": {
        "name": "Case 3: Real Betis vs Panathinaikos (2025 UEFA Europa League)",
        "script": BASE_DIR / "cases" / "03_real_betis_vs_panathinaikos_2025" / "main.py",
        "description": "Object-oriented architecture (PitchLineDetector, TeamDetector, OffsideLineDrawer) with high-res vanishing point calibration."
    }
}


def run_case(case_id: str, no_gui: bool = False):
    case_info = CASES.get(case_id)
    if not case_info:
        print(f"[!] Error: Invalid case ID '{case_id}'. Choose from: {list(CASES.keys())}")
        return False

    script_path = case_info["script"]
    if not script_path.exists():
        print(f"[!] Error: Script not found at {script_path}")
        return False

    print("\n" + "=" * 70)
    print(f" ▶ {case_info['name']}")
    print(f"   {case_info['description']}")
    print("=" * 70)

    cmd = [sys.executable, str(script_path)]
    if no_gui:
        cmd.append("--no-gui")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Runner for Football Offside Line Detection & Perspective Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --case 1
  python run_demo.py --case 3
  python run_demo.py --all
  python run_demo.py --case 2 --no-gui
        """
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["1", "2", "3"],
        help="Select a specific match case study to run (1, 2, or 3)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three match case studies sequentially"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without displaying OpenCV imshow GUI windows (headless-safe)"
    )

    args = parser.parse_args()

    if not args.case and not args.all:
        print("\n" + "=" * 70)
        print(" Football Offside Detection - Available Case Studies")
        print("=" * 70)
        for cid, info in CASES.items():
            print(f" [{cid}] {info['name']}")
            print(f"     {info['description']}\n")
        print("Usage: python run_demo.py --case <1|2|3> [--no-gui]")
        print("       python run_demo.py --all [--no-gui]")
        sys.exit(0)

    if args.all:
        for cid in sorted(CASES.keys()):
            success = run_case(cid, no_gui=args.no_gui)
            if not success:
                print(f"[!] Warning: Case {cid} did not finish cleanly.")
    elif args.case:
        run_case(args.case, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
