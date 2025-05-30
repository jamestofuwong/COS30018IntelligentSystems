import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def parse_yolo_annotations(file_path):
    boxes = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_center, y_center, width, height = map(float, parts)
                boxes.append((cls, x_center, y_center, width, height))
    return boxes

def convert_yolo_to_bbox(box, img_width, img_height):
    cls, x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return cls, x1, y1, x2, y2

def evaluate_for_threshold(test_folder, predictions_folder, img_width, img_height):
    iou_thresholds = [0.5, 0.7]

    precisions = []
    recalls = []
    f1_scores = []
    mAP50s = []
    mAP70s = []

    test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.txt')])
    pred_files_set = set([f for f in os.listdir(predictions_folder) if f.endswith('.txt')])

    for filename in tqdm(test_files):
        test_file = os.path.join(test_folder, filename)
        prediction_file = os.path.join(predictions_folder, filename)

        if filename not in pred_files_set:
            print(f"Warning: Prediction file not found for {filename}. Skipping this file.")
            continue

        test_boxes = parse_yolo_annotations(test_file)
        pred_boxes = parse_yolo_annotations(prediction_file)

        test_bboxes = [convert_yolo_to_bbox(box, img_width, img_height) for box in test_boxes]
        pred_bboxes = [convert_yolo_to_bbox(box, img_width, img_height) for box in pred_boxes]

        matched_preds = set()
        matched_gts = set()

        # Corrected matching logic: one-to-one matching based on best IoU >= threshold
        for i, gt_box in enumerate(test_bboxes):
            best_iou = 0
            best_j = -1
            for j, pred_box in enumerate(pred_bboxes):
                if j in matched_preds:
                    continue  # prediction already matched

                iou = calculate_iou(gt_box[1:], pred_box[1:])
                if iou >= iou_thresholds[0] and iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                matched_gts.add(i)
                matched_preds.add(best_j)

        tp = len(matched_preds)
        fp = len(pred_bboxes) - tp
        fn = len(test_bboxes) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        ap_results = {}
        for threshold in iou_thresholds:
            ious_for_threshold = []
            for i, gt_box in enumerate(test_bboxes):
                max_iou = 0
                for pred_box in pred_bboxes:
                    iou = calculate_iou(gt_box[1:], pred_box[1:])
                    if iou >= threshold and iou > max_iou:
                        max_iou = iou
                ious_for_threshold.append(max_iou)
            ap = sum(iou >= threshold for iou in ious_for_threshold) / len(ious_for_threshold) if ious_for_threshold else 0
            ap_results[threshold] = ap

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        mAP50s.append(ap_results.get(0.5, 0))
        mAP70s.append(ap_results.get(0.7, 0))

    avg_metrics = {
        'Precision': np.mean(precisions) if precisions else 0,
        'Recall': np.mean(recalls) if recalls else 0,
        'F1-Score': np.mean(f1_scores) if f1_scores else 0,
        'mAP50': np.mean(mAP50s) if mAP50s else 0,
        'mAP70': np.mean(mAP70s) if mAP70s else 0
    }
    return avg_metrics

def evaluate_all_thresholds(test_folder, base_predictions_folder, img_width, img_height, output_csv):
    thresholds = [d for d in os.listdir(base_predictions_folder) if os.path.isdir(os.path.join(base_predictions_folder, d))]
    thresholds = sorted(thresholds, key=lambda x: float(x))  # sort by numeric threshold value

    results = []
    for threshold in thresholds:
        pred_folder = os.path.join(base_predictions_folder, threshold)
        print(f"Evaluating threshold: {threshold}")
        avg_metrics = evaluate_for_threshold(test_folder, pred_folder, img_width, img_height)
        avg_metrics['Confidence_Threshold'] = float(threshold)
        results.append(avg_metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[['Confidence_Threshold', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP70']]
    results_df.to_csv(output_csv, index=False)
    print(f"All thresholds evaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    test_folder = "./platedetection-test-dataset/labels"
    base_predictions_folder = "./both-models-results"  # contains subfolders named by confidence thresholds
    img_width, img_height = 1920, 1080
    output_csv = "./evaluations/bothmodels.csv"

    evaluate_all_thresholds(test_folder, base_predictions_folder, img_width, img_height, output_csv)
