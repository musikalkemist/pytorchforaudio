"""
UrbanSound8K Dataset Setup Utility
----------------------------------
This script automates the acquisition and organization of the UrbanSound8K dataset 
for the TSOAI (The Sound of AI) courses. 

Functionality:
1. Downloads the latest dataset version from Kaggle via kagglehub.
2. Sanitizes configuration paths by removing parent-directory relative markers 
   to ensure the dataset is housed within the project root.
3. Relocates the metadata CSV into the local 'metadata' directory.
4. Moves or copies the high-volume audio directory from the system cache 
   to the local project environment while preventing directory nesting errors.
5. Supports both 'Copy' (preserves cache) and 'Move' (disk-efficient) modes.

Usage: 
    python dataset_downloader.py         # Defaults to MOVE mode
    python dataset_downloader.py --COPY  # Activates COPY mode
"""

import sys
import os
import shutil
import kagglehub
import argparse

path_config = "10 Predictions with sound classifier" # Load dataset paths from 'urbansounddataset.py' file
folder_path = os.path.join(os.getcwd(), path_config) # Get the current directory and join it with the folder name
sys.path.append(folder_path) if folder_path not in sys.path else None # Add this path to Python's search list
from urbansounddataset import DATASET_PATH, AUDIO_DIR, ANNOTATIONS_FILE

# 0. Setup Argument Parser
parser = argparse.ArgumentParser(description="Download and organize UrbanSound8K dataset.")
parser.add_argument('--COPY', action='store_true', help="Copy dataset from Kaggle folder instead of moving files.")
args = parser.parse_args()
COPY_MODE = args.COPY # OPTIONAL: 'True' for keeping Kaggle cache

print()
print("="*80)
print("Starting automated dataset downloader script for UrbanSound8K for TSOAI courses!")
print(f"After download ends, all files will be {'copied' if COPY_MODE else 'moved'} into destination folder.")
print("="*80)

# 1. Downloads latest dataset version
dataset_config = "chrisfilo/urbansound8k"
print(f"\n1️⃣  Downloading dataset from '{dataset_config}'…\n⏳ Please wait until it's finished, it could take a while :)")
DOWNLOAD_PATH = kagglehub.dataset_download(dataset_config)
print("☑️  Completed dataset download into:\n  ", DOWNLOAD_PATH)

# 2. Imports destination paths
print(f"\n2️⃣  Loading destination dataset paths from '{path_config}/urbansounddataset.py' …")
AUDIO_DIR, ANNOTATIONS_FILE, DATASET_PATH = [path.removeprefix("../") # Batch cleaning paths
                                             for path in [AUDIO_DIR, ANNOTATIONS_FILE, DATASET_PATH]]
print("☑️  Loaded paths:", AUDIO_DIR, ANNOTATIONS_FILE, DATASET_PATH, sep="\n   ", end="\n")

# 3. Moves annotations file (CSV)
filename = 'UrbanSound8K.csv'
origin = f"{DOWNLOAD_PATH}/{filename}"
destination = ANNOTATIONS_FILE
print(f"\n3️⃣  {'Copying' if COPY_MODE else 'Moving'} metadata {filename}:\n  {DOWNLOAD_PATH}\n   -> {DATASET_PATH}")
try:
    os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True) # Verifies folder existance
    shutil.copy(origin, destination) # Copy downloaded CSV file
    if not COPY_MODE: os.remove(origin) # Mimics a 'move'
    print(f"☑️  Annotations file successfully {'copied' if COPY_MODE else 'moved'} into '{DATASET_PATH}'")
except Exception as e:
    print(f"⚠️ Unexpected Error when processing folder '{DOWNLOAD_PATH}':\n{e}")
    print(f"Please do it manually. Destination: '{DATASET_PATH}'")

# 4. Moves all audio files
print(f"\n4️⃣  {'Copying' if COPY_MODE else 'Moving'} audio files:\n  {DOWNLOAD_PATH}\n   -> {AUDIO_DIR}")
try:
    shutil.copytree(DOWNLOAD_PATH, AUDIO_DIR, dirs_exist_ok=True) # Copy downloaded audio files
    if not COPY_MODE: shutil.rmtree(DOWNLOAD_PATH) # Mimics a 'move' without nesting errors
    print(f"☑️  Audio files succesfully {'copied' if COPY_MODE else 'moved'} into '{AUDIO_DIR}'")
except Exception as e:
    print(f"⚠️ Unexpected Error when processing folder '{DOWNLOAD_PATH}':\n{e}")
    print(f"Please do it manually. Destination: '{AUDIO_DIR}'")
finally:
    # Remove redundant CSV if it was accidentally copied into the audio folder
    redundant_csv = os.path.join(AUDIO_DIR, 'UrbanSound8K.csv')
    if COPY_MODE and os.path.exists(redundant_csv):
        os.remove(redundant_csv)

print("="*80)
print("✅ Success! Dataset is ready for training.\n")