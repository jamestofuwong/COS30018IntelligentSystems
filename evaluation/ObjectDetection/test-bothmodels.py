import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm

def main():
    # Configuration
    input_folder = './platedetection-test-dataset/images'              # Folder with input images
    output_annotation_dir = './both-models-results/0.8'                # Folder to save plate annotation files
    vehicle_model_path = './vehicle.pt'
    plate_model_path = './plate.pt'
    vehicle_conf_threshold = 0.8
    plate_conf_threshold = 0.8

    # Ensure output directory exists
    os.makedirs(output_annotation_dir, exist_ok=True)

    # Load YOLO models
    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)

    # List image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(input_folder, filename)

        # Read image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Warning: Could not load image file {input_image_path}. Skipping...")
            continue

        height, width, _ = image.shape

        # Vehicle detection
        vehicle_results = vehicle_model(image, conf=vehicle_conf_threshold, verbose=False)

        plate_annotation_lines = []

        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop vehicle ROI for plate detection
                vehicle_roi = image[y1:y2, x1:x2]
                if vehicle_roi.size == 0:
                    continue

                # Plate detection inside vehicle ROI
                plate_results = plate_model(vehicle_roi, conf=plate_conf_threshold, verbose=False)

                for plate_result in plate_results:
                    for plate_box in plate_result.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_conf = float(plate_box.conf[0])
                        cls_plate = int(plate_box.cls[0])  # Assuming class index for plate

                        # Convert plate coords to absolute image coords
                        abs_px1 = x1 + px1
                        abs_py1 = y1 + py1
                        abs_px2 = x1 + px2
                        abs_py2 = y1 + py2

                        # Save plate detection annotation in YOLO format
                        x_center_p = ((abs_px1 + abs_px2) / 2) / width
                        y_center_p = ((abs_py1 + abs_py2) / 2) / height
                        box_width_p = (abs_px2 - abs_px1) / width
                        box_height_p = (abs_py2 - abs_py1) / height

                        plate_annotation_lines.append(f"{cls_plate} {x_center_p:.6f} {y_center_p:.6f} {box_width_p:.6f} {box_height_p:.6f}")

        # Write plate annotations to file
        plate_annotation_path = os.path.join(output_annotation_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(plate_annotation_path, 'w') as pf:
            pf.write('\n'.join(plate_annotation_lines))

    print("Processing complete. Plate annotations saved.")

if __name__ == "__main__":
    main()
