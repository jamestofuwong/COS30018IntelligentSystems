import csv
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance

def load_csv(file_path):
    """Load the CSV file into a dictionary with file_name as the key and label as the value."""
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:  # Ensure row has at least two columns
                data[row[0]] = row[1].strip()
    return data

def calculate_metrics(ground_truth, predictions):
    """Calculate various evaluation metrics."""
    exact_match_count = 0
    total_characters = 0
    correct_characters = 0
    total_levenshtein_distance = 0
    partial_match_count = 0
    no_match_count = 0
    total_entries = len(ground_truth)

    for file_name, true_label in ground_truth.items():
        predicted_label = predictions.get(file_name, "")

        # Exact match
        if true_label == predicted_label:
            exact_match_count += 1

        # Character accuracy
        matcher = SequenceMatcher(None, true_label, predicted_label)
        correct_characters += sum(triple[-1] for triple in matcher.get_matching_blocks())
        total_characters += len(true_label)

        # Levenshtein distance
        total_levenshtein_distance += levenshtein_distance(true_label, predicted_label)

        # Partial match
        if predicted_label in true_label or true_label in predicted_label:
            partial_match_count += 1

        # No match
        if predicted_label == "":
            no_match_count += 1

    exact_match_rate = exact_match_count / total_entries
    character_accuracy = correct_characters / total_characters if total_characters else 0
    average_levenshtein_distance = total_levenshtein_distance / total_entries
    word_error_rate = total_levenshtein_distance / total_characters if total_characters else 0
    partial_match_rate = partial_match_count / total_entries
    no_match_rate = no_match_count / total_entries

    return {
        "Exact Match Rate": exact_match_rate,
        "Character Accuracy": character_accuracy,
        "Word Error Rate": word_error_rate,
        "Average Levenshtein Distance": average_levenshtein_distance,
        "Partial Match Rate": partial_match_rate,
        "No Match Rate": no_match_rate
    }

def save_metrics_to_csv(metrics, output_file):
    """Save the metrics to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for metric, value in metrics.items():
            writer.writerow([metric, f"{value:.4f}"])

if __name__ == "__main__":
    # File paths for the ground truth and prediction CSV files
    ground_truth_csv = "./platerecognition-test-dataset/label.csv"
    predictions_csv = "./OCR-results/custom-trained-results.csv"
    output_metrics_csv = "./evaluations/customOCR.csv"

    # Load data
    ground_truth = load_csv(ground_truth_csv)
    predictions = load_csv(predictions_csv)

    # Ensure labels are compared only for matching file names
    ground_truth = {k: v for k, v in ground_truth.items() if k in predictions}
    predictions = {k: v for k, v in predictions.items() if k in ground_truth}

    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)

    # Save metrics to a CSV file
    save_metrics_to_csv(metrics, output_metrics_csv)

    print(f"Metrics saved to '{output_metrics_csv}'.")
