"""
Download helper for Camargo 2021 and Reznick 2021 benchmark datasets.

Camargo 2021:
  "A comprehensive, open-source dataset of lower limb biomechanics in multiple
   conditions of stairs, ramps, and level-ground ambulation and transitions."
  - 22 able-bodied subjects, .mat format
  - IMU (trunk/thigh/shank/foot), EMG, Goniometers, Motion Capture
  - Conditions: level ground, ramps (5-18 deg), stairs, transitions
  - Mendeley Data DOI: 10.17632/fcgm3chfff.2
  - EPIC Lab mirror: http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/

Reznick 2021:
  "Lower-limb kinematics and kinetics during continuously varying human locomotion."
  - 10 able-bodied subjects
  - Vicon motion capture + Bertec treadmill
  - Walking (variable speed/incline), running, stairs, transitions
  - DOI: 10.1038/s41597-021-01057-9

Both datasets are large (several GB). This script provides download URLs
and verifies the directory structure after manual download.
"""

import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMARGO_DIR = os.path.join(PROJECT_ROOT, "data", "camargo2021")
REZNICK_DIR = os.path.join(PROJECT_ROOT, "data", "reznick2021")

CAMARGO_URLS = {
    "mendeley_part1": "https://data.mendeley.com/datasets/fcgm3chfff/2",
    "mendeley_part2": "https://data.mendeley.com/datasets/k9gm4kz7dr/2",
    "mendeley_part3": "https://data.mendeley.com/datasets/czxs6hf946/2",
    "epic_lab_mirror": "http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/",
}

REZNICK_URLS = {
    "figshare": "https://springernature.figshare.com/collections/Lower-limb_Kinematics_and_Kinetics_During_Continuously_Varying_Human_Locomotion/5175254/1",
    "paper": "https://doi.org/10.1038/s41597-021-01057-9",
}


def check_camargo():
    """Check if Camargo 2021 dataset is present and report structure."""
    print("=" * 60)
    print("CAMARGO 2021 DATASET")
    print("=" * 60)

    if not os.path.exists(CAMARGO_DIR):
        os.makedirs(CAMARGO_DIR, exist_ok=True)
        print(f"[MISSING] Created empty directory: {CAMARGO_DIR}")
        print("\nDownload instructions:")
        print("  1. Visit the EPIC Lab page (recommended, has Dropbox mirror):")
        print(f"     {CAMARGO_URLS['epic_lab_mirror']}")
        print("  2. Or download all 3 parts from Mendeley Data:")
        for name, url in CAMARGO_URLS.items():
            if name.startswith("mendeley"):
                print(f"     {name}: {url}")
        print(f"  3. Extract .mat files into: {CAMARGO_DIR}/")
        print("     Expected structure: data/camargo2021/AB01/, AB02/, ..., AB22/")
        return False

    mat_files = glob.glob(os.path.join(CAMARGO_DIR, "**", "*.mat"), recursive=True)
    subject_dirs = [
        d for d in os.listdir(CAMARGO_DIR)
        if os.path.isdir(os.path.join(CAMARGO_DIR, d)) and d.startswith("AB")
    ]

    if mat_files:
        print(f"[OK] Found {len(mat_files)} .mat files across {len(subject_dirs)} subjects")
        print(f"  Subjects: {sorted(subject_dirs)}")
        return True
    else:
        print(f"[PARTIAL] Directory exists but no .mat files found")
        print(f"  Contents: {os.listdir(CAMARGO_DIR)[:10]}")
        return False


def check_reznick():
    """Check if Reznick 2021 dataset is present and report structure."""
    print("\n" + "=" * 60)
    print("REZNICK 2021 DATASET")
    print("=" * 60)

    if not os.path.exists(REZNICK_DIR):
        os.makedirs(REZNICK_DIR, exist_ok=True)
        print(f"[MISSING] Created empty directory: {REZNICK_DIR}")
        print("\nDownload instructions:")
        print("  1. Visit the Figshare collection:")
        print(f"     {REZNICK_URLS['figshare']}")
        print("  2. Read the data descriptor paper for format details:")
        print(f"     {REZNICK_URLS['paper']}")
        print(f"  3. Extract files into: {REZNICK_DIR}/")
        return False

    all_files = glob.glob(os.path.join(REZNICK_DIR, "**", "*"), recursive=True)
    data_files = [f for f in all_files if f.endswith((".mat", ".csv", ".h5"))]

    if data_files:
        print(f"[OK] Found {len(data_files)} data files")
        extensions = set(os.path.splitext(f)[1] for f in data_files)
        print(f"  File types: {extensions}")
        return True
    else:
        print(f"[PARTIAL] Directory exists but no data files found")
        return False


def main():
    print("Checking benchmark dataset availability...\n")

    cam_ok = check_camargo()
    rez_ok = check_reznick()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Camargo 2021: {'READY' if cam_ok else 'NEEDS DOWNLOAD'}")
    print(f"  Reznick 2021: {'READY' if rez_ok else 'NEEDS DOWNLOAD'}")

    if not (cam_ok and rez_ok):
        print("\nNote: The Camargo dataset is the primary training dataset.")
        print("You can start development with Camargo alone; Reznick is used")
        print("for phase prediction validation (Phase 2+).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
