import numpy as np
from ultralytics import YOLO
from constants import dataset_yaml_path, yolo_model_path, data_root
import torch
from torch.utils.data import DataLoader
from yolo_aroi.f_testing import YOLOtoSegmentationDataset, evaluate_model, evaluate_model_box

if __name__ == "__main__":
    model = YOLO(yolo_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = YOLOtoSegmentationDataset("../" + data_root + "yolo/", image_size=512, split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # # Evaluate the model
    # metrics = evaluate_model(model, dataset_yaml_path, test_loader, device)
    #
    # # Print metrics in the same format as the UNet evaluation
    # print("Evaluation Metrics:")
    # print(f"mAP@50: {metrics['mAP@50']:.3f}")
    # print(f"mAP@50-95: {metrics['mAP@50-95']:.3f}")
    # print(f"Mean IoU: {metrics['Mean IoU']:.3f}")
    # print(f"Mean F1: {metrics['Mean F1']:.3f}")
    # print(f"Mean Precision: {metrics['Mean Precision']:.3f}")
    # print(f"Mean Recall: {metrics['Mean Recall']:.3f}")

    metrics = evaluate_model_box(model, dataset_yaml_path, test_loader, device, conf=0.25)

    # Print metrics in the same format as the UNet evaluation
    print("Evaluation Metrics:")
    print(f"mAP@50:         {metrics["Box mAP@50"   ]:.3f}")
    print(f"mAP@50-95:      {metrics["Box mAP@50-95"]:.3f}")
    print(f"Mean F1:        {metrics["Box F1"]:.3f}")
    print(f"Mean Precision: {metrics["Box Precision"  ]:.3f}")
    print(f"Mean Recall:    {metrics["Box Recall"     ]:.3f}")


    # Visualize some predictions
    # visualize_predictions(model, test_loader, device)
    # visualize_predictions_with_ground_truth(model, test_loader, device)