"""
Run the 2D processing pipeline for one case, from segmentation to the sampled tracks.
"""

import logging
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROC = os.path.join(SCRIPT_DIR, "data_process")

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True,
                    help="Parent dir containing {case_name}/{color,depth,calibrate.pkl,metadata.json}.")
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--category", type=str, required=True,
                    help="Object category used as GroundingDINO TEXT_PROMPT base (e.g. 'yellow towel').")
parser.add_argument("--controller", type=str, default="hand",
                    help="Manipulator name; appended to TEXT_PROMPT and used to label controller "
                         "masks (e.g. 'hand' for a human, 'robot gripper' for a robot arm).")
parser.add_argument("--headless", action="store_true",
                    help="Render Open3D steps offscreen (no display, e.g. ssh/GPU node).")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
category = args.category
headless = ["--headless"] if args.headless else []
TEXT_PROMPT = f"{category}.{args.controller}"
CONTROLLER_NAME = args.controller


def setup_logger(log_file="timer.log"):
    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


logger = setup_logger()


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start = time.time()
        logger.info("[%s] %s: start", case_name, self.task_name)

    def __exit__(self, *exc):
        logger.info(
            "[%s] %s: done in %.2f s", case_name, self.task_name, time.time() - self.start
        )


def run(script, *args_):
    cmd = [sys.executable, os.path.join(DATA_PROC, script), *args_]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"step failed (rc={rc}): {' '.join(cmd)}")


with Timer("Video Segmentation"):
    run("segment.py",
        "--base_path", base_path,
        "--case_name", case_name,
        "--TEXT_PROMPT", TEXT_PROMPT)

with Timer("Dense Tracking"):
    run("dense_track.py",
        "--base_path", base_path,
        "--case_name", case_name)

with Timer("Lift to 3D"):
    run("data_process_pcd.py",
        "--base_path", base_path,
        "--case_name", case_name,
        *headless)

with Timer("Mask Post-Processing"):
    run("data_process_mask.py",
        "--base_path", base_path,
        "--case_name", case_name,
        "--controller_name", CONTROLLER_NAME,
        *headless)

with Timer("Data Tracking"):
    run("data_process_track.py",
        "--base_path", base_path,
        "--case_name", case_name,
        "--controller", CONTROLLER_NAME,
        *headless)

with Timer("Final Data Generation"):
    run("data_process_sample.py",
        "--base_path", base_path,
        "--case_name", case_name)
