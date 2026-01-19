# PytorchForAudio
Code for the "[PyTorch for Audio + Music Processing](https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm)" series on The Sound of AI YouTube channel.

This repository is a comprehensive collection of resources and code for understanding and implementing deep learning models for audio tasks using PyTorch and Torchaudio. It serves as a practical guide, moving from basic neural network implementations to building a complete sound classification system (CNN) trained on the UrbanSound8K dataset.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Torchaudio](https://img.shields.io/badge/Torchaudio-black?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### Note on Versioning
> While this v2 release is fully functional and optimized for current environments, it may differ from the original version shown in the course. The codebase has been updated to reflect modern best practices and improved dependency management. Consequently, the original course version has been deprecated; however, it remains available in the [legacy branch](https://github.com/musikalkemist/pytorchforaudio/tree/legacy) for those wishing to follow the video content exactly.

# Table of Contents
* [Dataset Setup (UrbanSound8K)](#dataset-setup-urbansound8k)
* [Course Structure](#course-structure)
    * [1. Introduction & Basics](#introduction--basics)
    * [2. Audio Data Processing](#audio-data-processing)
    * [3. Sound Classification Project](#sound-classification-project-urbansound8k)
* [How to Run the Scripts](#how-to-run-the-scripts)

---

## Dataset Setup _(UrbanSound8K)_

To run the sound classification lessons (8-10), you will need the UrbanSound8K dataset. We provide an **automated downloader** to handle the acquisition, path sanitization, and folder organization for you.
* **Quick Start:** Run `python dataset_downloader.py` from the root directory.
* **Options:** Supports `--COPY` flag to preserve your Kaggle cache.
> **Full Instructions:** Please check the [Instructions UrbanSound8K](Instructions_UrbanSound8K.md) file for help using the downloader script or manual download steps.

## Course Structure

### Introduction & Basics

1.  **Course Overview:** _[Video][1yt] | [Slides][1sl]_
2.  **Implementing and Training a Neural Network:** _[Video][2yt] | [Code][2cd]_
3.  **Making Predictions with PyTorch Models:** _[Video][3yt] | [Code][3cd]_

---

### Audio Data Processing

4.  **Custom Audio PyTorch Dataset:** _[Video][4yt] | [Code][4cd]_
5.  **Extracting Mel Spectrograms:** _[Video][5yt] | [Code][5cd]_
6.  **Pre-processing Audio (Padding/Truncating):** _[Video][6yt] | [Code][6cd]_
7.  **Pre-processing on GPU:** _[Video][7yt] | [Code][7cd]_

---

### Sound Classification Project (UrbanSound8K)

8.  **Implementing a CNN for Sound Classification:** _[Video][8yt] | [Code][8cd]_
9.  **Training a Sound Classifier:** _[Video][9yt] | [Code][9cd]_
10. **Predictions with a Sound Classifier:** _[Video][10yt] | [Code][10cd]_

---

## How to Run the Scripts
To ensure the models and scripts execute correctly, please follow these steps from your terminal:

### 1. Prepare the Environment (Recommended)

Before running inference, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Navigate to the Lesson Folder

Each class is self-contained. Move into the specific directory for the lesson you are studying:

```bash
cd 'class/folder/name'  # Replace with the specific directory path (ensure it is enclosed in quotes).
```

### 3. Execute the Script

Run the inference or training script using Python:

```bash
python inference.py  # Replace with the specific script name
```


<!-- Reference links for every chapter:
YouTube videos (#yt), PDF-file slides (#sl) and Jupyter Notebooks (#nb) -->
[1yt]: https://www.youtube.com/watch?v=gp2wZqDoJ1Y
[1sl]: <01 Course overview/PyTorch for Audio and Music Processing.pdf>

[2yt]: https://www.youtube.com/watch?v=4p0G6tgNLis
[2cd]: <02 Training a feed forward network/train.py>

[3yt]: https://www.youtube.com/watch?v=0Q5KTt2R5w4
[3cd]: <03 Making predictions/inference.py>

[4yt]: https://www.youtube.com/watch?v=88FFnqt5MNI
[4cd]: <04 Creating a custom dataset/urbansounddataset.py>

[5yt]: https://www.youtube.com/watch?v=lhF_RVa7DLE
[5cd]: <05 Extracting Mel spectrograms/urbansounddataset.py>

[6yt]: https://www.youtube.com/watch?v=WyJvrzVNkOc
[6cd]: <06 Padding audio files/urbansounddataset.py>

[7yt]: https://www.youtube.com/watch?v=3wD_eocmeXA
[7cd]: <07 Preprocessing data on GPU/urbansounddataset.py>

[8yt]: https://www.youtube.com/watch?v=SQ1iIKs190Q
[8cd]: <08 Implementing a CNN network/cnn.py>

[9yt]: https://www.youtube.com/watch?v=MMkeLjcBTcI
[9cd]: <09 Training urban sound classifier/train.py>

[10yt]: https://www.youtube.com/watch?v=ZeBvt1y237k
[10cd]: <10 Predictions with sound classifier/inference.py>