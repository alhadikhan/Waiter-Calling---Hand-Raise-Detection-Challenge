# Waiter Calling - Hand Raise Detection Challenge

This repository contains a Python script to process a provided video and detect at which desk a hand is being raised using YOLOv8. The project involves extracting frames from the video, creating a dataset, training a YOLOv8 model, and performing inference to identify hand raises in the video.

## Challenge Description

Write a Python script to process the provided video in such a way that:

- **Detect hand raised:** Process the provided video using Python to detect at which desk the hand is being raised.

### Details

1. Watch the [video](https://ml-hiring.fringecore.sh/waiter_calling/desk_video.mp4) for more information.
2. Solve the challenge in a single Python script.
3. Provide a `requirements.txt` if any libraries are required.

### Resources

1. [Video](https://ml-hiring.fringecore.sh/waiter_calling/desk_video.mp4)
2. [Person_name image](https://ml-hiring.fringecore.sh/waiter_calling/IMG.png)

### Partial Evaluation Criteria

- You are able to detect all hands raised.
- You are able to find which person is raising their hand.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Place your input video in the same directory as `main.py` and name it `video.mp4` (or update the script with your video path).

2. Run the script:
    ```bash
    python main.py
    ```
### Solution Steps

1. Extract Frames from Video: The script extracts frames from the provided video and saves them 
   in the extracted_frames directory.

2. Prepare YOLOv8 Dataset:

   Define class names for different people raising their hands.

   Specify frame ranges and bounding boxes for each person.

   Generate YOLO labels for each frame based on the specified annotations.

   Copy images to training and validation directories.

3. Create YAML Configuration: The script creates a YAML file for the YOLOv8 dataset 
   configuration, specifying paths for training and validation images, the number of classes, 
   and class names.

4. Train YOLOv8 Model: The script trains a YOLOv8 model using the prepared dataset, saving the 
   best model weights.

5. Run Inference on Video: The script loads the trained YOLOv8 model and performs inference on 
   the video, saving the output video with detected hand raises.

### Project Stucture
|-- main.py                # Main script containing all steps
|-- requirements.txt       # List of required libraries
|-- README.md              # Project description and instructions
|-- desk_video.mp4         # Input video file
|-- extracted_frames/      # Directory for extracted frames
|-- dataset/               # Directory for YOLO formatted dataset
