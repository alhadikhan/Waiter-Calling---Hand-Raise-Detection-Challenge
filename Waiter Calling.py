#This was done for image extraction

# import cv2
# import os

# # Specify the path to your video file
# video_path = "/kaggle/input/hand-raise-detection-1/desk_video.mp4"
# output_dir = "extracted_frames"
# os.makedirs(output_dir, exist_ok=True)

# # Open the video file
# vidcap = cv2.VideoCapture(video_path)
# success, image = vidcap.read()
# count = 0

# # Extract and save frames without resizing
# while success:
#     frame_path = os.path.join(output_dir, f"frame{count:04d}.jpg")
#     cv2.imwrite(frame_path, image)
#     success, image = vidcap.read()
#     print(f'Saved frame {count}')
#     count += 1

# vidcap.release()
# print("Finished extracting frames.") 



#This was for creating a perfect dataset to train YOLO Model

# import os
# import glob
# import cv2
# import shutil
# import random

# # Define class names (one per person raising their hand)
# class_names = {
#     "Tanvir": 0,
#     "Anik": 1,
#     "Toufik": 2,
#     "Imran": 3,
#     "Mufrad": 4,
#     "Emon": 5,
#     "Shafayet": 6,
#     "Faisal": 7
# }

# # Define annotations as you provided
# annotations = {
#     "Tanvir": {"frames": (95, 203), "bbox": (538, 445, 155, 155)},
#     "Anik": {"frames": (234, 305), "bbox": (919, 438, 350, 350)},
#     "Toufik": {"frames": (332, 436), "bbox": (785, 427, 187, 179)},
#     "Imran": {"frames": (426, 495), "bbox": (1112, 462, 183, 184)},
#     "Mufrad": {"frames": (589, 694), "bbox": (946, 435, 182, 195)},
#     "Emon": {"frames": (713, 761), "bbox": (1191, 476, 210, 211)},
#     "Shafayet": {"frames": (785, 865), "bbox": (686, 453, 175, 189)},
#     "Faisal": {"frames": (910, 988), "bbox": (651, 438, 266, 291)}
# }

# # Define the directory for images
# image_directory = '/kaggle/input/extracted-frames/'

# # Directory to save YOLO formatted labels and images
# dataset_directory = '/kaggle/working/dataset/'

# # Create the necessary subdirectories
# os.makedirs(os.path.join(dataset_directory, 'images/train'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'images/val'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'labels/train'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'labels/val'), exist_ok=True)

# # Function to generate YOLO labels for each frame
# def generate_yolo_labels(image_file, annotations, label_directory):
#     # Get the frame number from the image filename
#     frame_number = int(image_file.split('/')[-1].split('frame')[1].split('.jpg')[0])
    
#     # Create the label file path
#     label_file = os.path.join(label_directory, f"frame{frame_number:04d}.txt")
    
#     with open(label_file, 'w') as label_f:
#         labels_written = False
#         for person, details in annotations.items():
#             start_frame, end_frame = details["frames"]
#             x, y, w, h = details["bbox"]
#             if start_frame <= frame_number <= end_frame:
#                 image = cv2.imread(image_file)
#                 if image is None:
#                     print(f"Failed to read image: {image_file}")
#                     continue
#                 img_height, img_width, _ = image.shape
#                 x_center = (x + w / 2) / img_width
#                 y_center = (y + h / 2) / img_height
#                 width = w / img_width
#                 height = h / img_height
                
#                 class_id = class_names[person]
#                 label_f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
#                 labels_written = True
        
#         if not labels_written:
#             os.remove(label_file)
#             print(f"No labels written for frame {frame_number}, deleting {label_file}")
#         else:
#             print(f"Labels written for frame {frame_number} at {label_file}")

# # Function to copy images to the respective directories
# def copy_images(image_file, dataset_directory):
#     frame_number = int(image_file.split('/')[-1].split('frame')[1].split('.jpg')[0])
    
#     subdir = 'train' if random.random() < 0.8 else 'val'
    
#     dest_path = os.path.join(dataset_directory, f'images/{subdir}', os.path.basename(image_file))
#     shutil.copy(image_file, dest_path)
#     print(f"Copied {image_file} to {dest_path}")
    
#     return dest_path

# # Process all images in the directory
# image_files = glob.glob(os.path.join(image_directory, '*.jpg'))

# missing_images = []

# # Generate labels and copy images
# for image_file in image_files:
#     copied_image_path = copy_images(image_file, dataset_directory)
#     generate_yolo_labels(copied_image_path, annotations, os.path.join(dataset_directory, 'labels', copied_image_path.split('/')[-2]))

# # Check for missing images based on annotations
# for person, details in annotations.items():
#     start_frame, end_frame = details["frames"]
#     for frame in range(start_frame, end_frame + 1):
#         expected_image_path = os.path.join(image_directory, f"frame{frame:04d}.jpg")
#         if not os.path.exists(expected_image_path):
#             missing_images.append(expected_image_path)

# if missing_images:
#     print(f"Missing images: {len(missing_images)}")
#     for missing_image in missing_images:
#         print(missing_image)
# else:
#     print("No missing images found.")

# # Count images with and without annotations
# train_image_dir = '/kaggle/working/dataset/images/train/'
# val_image_dir = '/kaggle/working/dataset/images/val/'

# raising_hand_count = 0
# not_raising_hand_count = 0

# def count_images_with_annotations(image_dir, label_dir):
#     global raising_hand_count, not_raising_hand_count

#     for image_file in os.listdir(image_dir):
#         if image_file.endswith('.jpg'):
#             label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
#             print(f"Checking label: {label_file}")  # Debug statement
#             if os.path.exists(label_file):
#                 with open(label_file, 'r') as f:
#                     if f.read().strip():
#                         raising_hand_count += 1
#                         print(f"Image with raising hand: {image_file}")
#                     else:
#                         not_raising_hand_count += 1
#             else:
#                 not_raising_hand_count += 1

# count_images_with_annotations(train_image_dir, '/kaggle/working/dataset/labels/train')
# count_images_with_annotations(val_image_dir, '/kaggle/working/dataset/labels/val')

# print(f"Images with raising hands: {raising_hand_count}")
# print(f"Images without raising hands: {not_raising_hand_count}")






#This was for Testing the dataset
# import os
# import glob
# import cv2
# import shutil
# import random

# # Define class names (one per person raising their hand)
# class_names = {
#     "Tanvir": 0,
#     "Anik": 1,
#     "Toufik": 2,
#     "Imran": 3,
#     "Mufrad": 4,
#     "Emon": 5,
#     "Shafayet": 6,
#     "Faisal": 7
# }

# # Define annotations as you provided
# annotations = {
#     "Tanvir": {"frames": (95, 203), "bbox": (538, 445, 155, 155)},
#     "Anik": {"frames": (234, 305), "bbox": (919, 438, 350, 350)},
#     "Toufik": {"frames": (332, 436), "bbox": (785, 427, 187, 179)},
#     "Imran": {"frames": (426, 495), "bbox": (1112, 462, 183, 184)},
#     "Mufrad": {"frames": (589, 694), "bbox": (946, 435, 182, 195)},
#     "Emon": {"frames": (713, 761), "bbox": (1191, 476, 210, 211)},
#     "Shafayet": {"frames": (785, 865), "bbox": (686, 453, 175, 189)},
#     "Faisal": {"frames": (910, 988), "bbox": (651, 438, 266, 291)}
# }

# # Define the directory for images
# image_directory = '/kaggle/input/extracted-frames/'

# # Directory to save YOLO formatted labels and images
# dataset_directory = '/kaggle/working/dataset/'

# # Create the necessary subdirectories
# os.makedirs(os.path.join(dataset_directory, 'images/train'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'images/val'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'labels/train'), exist_ok=True)
# os.makedirs(os.path.join(dataset_directory, 'labels/val'), exist_ok=True)

# # Function to generate YOLO labels for each frame
# def generate_yolo_labels(image_file, annotations, label_directory):
#     # Get the frame number from the image filename
#     frame_number = int(image_file.split('/')[-1].split('frame')[1].split('.jpg')[0])
    
#     # Create the label file path
#     label_file = os.path.join(label_directory, f"frame{frame_number:04d}.txt")
    
#     with open(label_file, 'w') as label_f:
#         labels_written = False
#         for person, details in annotations.items():
#             start_frame, end_frame = details["frames"]
#             x, y, w, h = details["bbox"]
#             if start_frame <= frame_number <= end_frame:
#                 image = cv2.imread(image_file)
#                 if image is None:
#                     print(f"Failed to read image: {image_file}")
#                     continue
#                 img_height, img_width, _ = image.shape
#                 x_center = (x + w / 2) / img_width
#                 y_center = (y + h / 2) / img_height
#                 width = w / img_width
#                 height = h / img_height
                
#                 class_id = class_names[person]
#                 label_f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
#                 labels_written = True
        
#         if not labels_written:
#             os.remove(label_file)
#             print(f"No labels written for frame {frame_number}, deleting {label_file}")
#         else:
#             print(f"Labels written for frame {frame_number} at {label_file}")

# # Function to copy images to the respective directories
# def copy_images(image_file, dataset_directory):
#     frame_number = int(image_file.split('/')[-1].split('frame')[1].split('.jpg')[0])
    
#     subdir = 'train' if random.random() < 0.8 else 'val'
    
#     dest_path = os.path.join(dataset_directory, f'images/{subdir}', os.path.basename(image_file))
#     shutil.copy(image_file, dest_path)
#     print(f"Copied {image_file} to {dest_path}")
    
#     return dest_path

# # Process all images in the directory
# image_files = glob.glob(os.path.join(image_directory, '*.jpg'))

# missing_images = []

# # Generate labels and copy images
# for image_file in image_files:
#     copied_image_path = copy_images(image_file, dataset_directory)
#     generate_yolo_labels(copied_image_path, annotations, os.path.join(dataset_directory, 'labels', copied_image_path.split('/')[-2]))

# # Check for missing images based on annotations
# for person, details in annotations.items():
#     start_frame, end_frame = details["frames"]
#     for frame in range(start_frame, end_frame + 1):
#         expected_image_path = os.path.join(image_directory, f"frame{frame:04d}.jpg")
#         if not os.path.exists(expected_image_path):
#             missing_images.append(expected_image_path)

# if missing_images:
#     print(f"Missing images: {len(missing_images)}")
#     for missing_image in missing_images:
#         print(missing_image)
# else:
#     print("No missing images found.")

# # Count images with and without annotations
# train_image_dir = '/kaggle/working/dataset/images/train/'
# val_image_dir = '/kaggle/working/dataset/images/val/'

# raising_hand_count = 0
# not_raising_hand_count = 0

# def count_images_with_annotations(image_dir, label_dir):
#     global raising_hand_count, not_raising_hand_count

#     for image_file in os.listdir(image_dir):
#         if image_file.endswith('.jpg'):
#             label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
#             print(f"Checking label: {label_file}")  # Debug statement
#             if os.path.exists(label_file):
#                 with open(label_file, 'r') as f:
#                     if f.read().strip():
#                         raising_hand_count += 1
#                         print(f"Image with raising hand: {image_file}")
#                     else:
#                         not_raising_hand_count += 1
#             else:
#                 not_raising_hand_count += 1

# count_images_with_annotations(train_image_dir, '/kaggle/working/dataset/labels/train')
# count_images_with_annotations(val_image_dir, '/kaggle/working/dataset/labels/val')

# print(f"Images with raising hands: {raising_hand_count}")
# print(f"Images without raising hands: {not_raising_hand_count}")



#This was for creating the Yaml file and ensuring the rootpath is given 

# import yaml

# # Define the data for the YAML file
# data = {
#     'path': '/kaggle/working/dataset',
#     'train': 'images/train',
#     'val': 'images/val',
#     'nc': 8,  # Number of classes
#     'names': {
#         0: 'Tanvir',
#         1: 'Anik',
#         2: 'Toufik',
#         3: 'Imran',
#         4: 'Mufrad',
#         5: 'Emon',
#         6: 'Shafayet',
#         7: 'Faisal'
#     }
# }

# # Specify the file path to save the YAML file
# yaml_file_path = '/kaggle/working/dataset/data.yaml'

# # Write data to the YAML file
# with open(yaml_file_path, 'w') as yaml_file:
#     yaml.dump(data, yaml_file, default_flow_style=False)

# print(f"YAML file created at: {yaml_file_path}")



#This was for model training and evaluation


# from ultralytics import YOLO

# # Define the path to your dataset YAML file
# yaml_path = '/kaggle/working/dataset/dataset.yaml'

# # Initialize the YOLOv8 model
# model = YOLO("yolov8n.yaml")  # You can also use other versions like 'yolov8s.yaml', 'yolov8m.yaml', etc.

# # Train the model with GPU (device=0 for the first GPU, use device='cuda' for any available GPU)
# model.train(
#     data=yaml_path, 
#     epochs=100, 
#     imgsz=640, 
#     batch=16, 
#     project='/kaggle/working/yolov8_output', 
#     name='hand_raise_model',
#     device='cuda'  # This will use the GPU
# )


from ultralytics import YOLO
import cv2

# Path to the video file
video_path = 'E:/Waiter Calling/desk_video.mp4'

# Load the trained YOLOv8 model
model = YOLO('E:/Waiter Calling/best.pt')  # Adjust the path to the best model weights

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video details (frame width, height, and FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output video
output_path = 'E:/Waiter Calling/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Perform inference on the frame using the YOLOv8 model
    results = model(frame)  # Perform inference

    # Check if there are detections
    if results:
        # Access the first result and plot predictions
        result = results[0]
        frame_with_predictions = result.plot()  # Use plot instead of render
    else:
        # If no detections, use the original frame
        frame_with_predictions = frame

    # Write the frame to the output video
    out.write(frame_with_predictions)

    # Display the frame with predictions
    cv2.imshow('Output Video', frame_with_predictions)

    # Wait for 1 ms and break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close the video display window
cv2.destroyAllWindows()

print(f"Prediction video saved to {output_path}")




