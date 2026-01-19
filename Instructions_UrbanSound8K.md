# How to Download the UrbanSound8K Dataset
This project requires the **UrbanSound8K** dataset. You can download it automatically using the provided Python script (recommended for TSOAI courses) or manually via Kaggle.

## Option 1: Automatic Download (Recommended)
This method uses the `dataset_downloader.py` script to fetch, sanitize paths, and organize the folders according to the project requirements.

### 1. Install Requirements
Ensure your virtual environment is active and install the necessary library:
```bash
pip install kagglehub
```

### 2. Run the Script
You can run the script in two modes depending on your disk space:
* **Move Mode (Default):** Moves files from the Kaggle cache to your project folder (efficient).
```bash
python dataset_downloader.py
```
* **OPTIONAL: Copy Mode:** Keeps the original files in the Kaggle cache and creates a copy in your project.
```bash
python dataset_downloader.py --COPY
```

### What this script does
1. **Downloads** the latest version of UrbanSound8K using `kagglehub`.
2. **Sanitizes Paths:** Automatically strips `../` prefixes from constants to keep the data inside the project root.
3. **Organizes Structure:** Creates `datasets/UrbanSound8K/`, moves the metadata CSV to `metadata/`, and relocates all audio folders.
4. **Prevents Nesting:** Ensures the dataset doesn't end up in a versioned subfolder (like `1/`).

---

## Option 2: Manual Download
If you prefer to manage files yourself, ensure the final structure matches the expectations of the classifier scripts.

### 1. Download from Kaggle
1. Visit: [https://www.kaggle.com/datasets/chrisfilo/urbansound8k](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
2. Download and unzip `archive.zip`.

### 2. Correct Directory Structure
Your project directory should look exactly like this for the code to run correctly:
```text
PytorchForAudio/
├── 10 Predictions with sound classifier/
│   ├── train.py
│   ├── urbansounddataset.py   <-- Config constants are here
│   └── ...
├── datasets/                 
│   └── UrbanSound8K/         
│       ├── audio/            <-- Contains fold1, fold2, etc.
│       └── metadata/
│           └── UrbanSound8K.csv
└── dataset_downloader.py
```

### 3. Path Configuration
The scripts now use a centralized configuration in `urbansounddataset.py`. If you change your folder names, update them there:
```python
# urbansounddataset.py
DATASET_PATH = "datasets/UrbanSound8K/"
ANNOTATIONS_FILE = f"{DATASET_PATH}metadata/UrbanSound8K.csv"
AUDIO_DIR = f"{DATASET_PATH}audio/"
```