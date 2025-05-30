import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm  # Import tqdm for progress bar

def main():
    # Configuration
    input_folder = './platedetection-test-dataset/images'  # Path to the folder containing input images
    output_annotations_path = './plate-only-results/0.8'  # Directory to save annotation text files
    model_path = './plate.pt'  # Path to your YOLOv8 model
    confidence_threshold = 0.8  # Confidence threshold for detection

    # Ensure output directory exists
    os.makedirs(output_annotations_path, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(model_path)

    # List all image files to process
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image in the input folder with progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(input_folder, filename)

        # Read the input image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not load image file {input_image_path}. Skipping...")
            continue

        height, width, _ = image.shape

        # Perform object detection
        results = model(image, conf=confidence_threshold, verbose=False)

        # Create the annotation file path
        base_filename = os.path.splitext(filename)[0]
        annotation_file_path = os.path.join(output_annotations_path, f"{base_filename}.txt")

        with open(annotation_file_path, 'w') as annotation_file:
            # Save YOLO format annotations
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Convert to YOLO format
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height

                    # Write annotation to file
                    annotation_file.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print("Processing complete.")

if __name__ == "__main__":
    main()
