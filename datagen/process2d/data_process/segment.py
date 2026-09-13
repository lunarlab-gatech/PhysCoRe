"""
Run SAM2 video segmentation for every camera in the case.
"""

import glob
import os
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_num = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num
print(f"Processing {case_name}")

video_script = os.path.join(SCRIPT_DIR, "segment_util_video.py")
for camera_idx in range(camera_num):
    print(f"Processing {case_name} camera {camera_idx}")
    rc = os.system(
        f'python "{video_script}" --base_path {base_path} --case_name {case_name} '
        f"--TEXT_PROMPT '{TEXT_PROMPT}' --camera_idx {camera_idx}"
    )
    if rc != 0:
        raise SystemExit(f"segment_util_video.py failed (rc={rc}) on camera {camera_idx}")
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
