import os
import pytesseract
from PIL import Image
import csv
from tqdm import tqdm  # For progress bar

# Configure Tesseract executable path if necessary
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def process_images_in_folder(folder_path, output_csv_path, char_list):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get all image files in the folder, sorted alphabetically (case insensitive)
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: x.lower()
    )

    if not image_files:
        print(f"No image files found in folder '{folder_path}'.")
        return

    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["file_name", "label"])

        # Initialize progress bar
        for image_file in tqdm(image_files, desc="Processing Images", unit="image"):
            image_path = os.path.join(folder_path, image_file)

            try:
                # Open the image
                image = Image.open(image_path)

                extracted_text = ""
                # Try both PSM modes: 7 for single-line and 6 for multi-line
                for psm_mode in [7, 6]:
                    tesseract_config = f"--psm {psm_mode} -c tessedit_char_whitelist={char_list}"

                    # Use Tesseract to extract text
                    text = pytesseract.image_to_string(image, config=tesseract_config).strip()

                    if text:
                        # Process multi-line text into a single line without spaces
                        extracted_text = "".join(text.splitlines()).replace(" ", "")
                        break

                # Write result to CSV (blank if no text detected)
                csv_writer.writerow([image_file, extracted_text])

            except Exception as e:
                print(f"Error processing file '{image_file}': {e}")
                # Still write a blank line on error to keep CSV in sync
                csv_writer.writerow([image_file, ""])

if __name__ == "__main__":
    # Folder containing the license plate images
    folder_path = "./platerecognition-test-dataset/images"

    # Output CSV file path
    output_csv_path = "./OCR-results/tesseract-results.csv"

    # Custom character list (e.g., alphanumeric characters for license plates)
    char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    process_images_in_folder(folder_path, output_csv_path, char_list)
