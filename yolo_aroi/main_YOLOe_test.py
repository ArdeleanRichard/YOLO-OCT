import numpy as np
from ultralytics import YOLO, RTDETR, YOLOE
from constants import dataset_yaml_path, data_root, yoloe_model_path, dataset_SEG_yaml_path
import torch
from torch.utils.data import DataLoader

from yolo_aroi.f_testing import YOLOtoSegmentationDataset, evaluate_model, evaluate_model_box, evaluate_model_box_mask

if __name__ == "__main__":
    model = YOLOE(yoloe_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = YOLOtoSegmentationDataset("../" + data_root + "saved_classes_all_sub_only_remap_SEG/", image_size=512, split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Evaluate the model
    metrics = evaluate_model_box_mask(model, dataset_SEG_yaml_path, test_loader, device)

    # Print metrics in the same format as the UNet evaluation
    print("Evaluation Metrics:")
    print(f"Box mAP@50:         {metrics["Box mAP@50"   ]:.3f}")
    print(f"Box mAP@50-95:      {metrics["Box mAP@50-95"]:.3f}")
    print(f"Box Mean F1:        {np.mean(np.array(metrics["Box F1"])):.3f}")
    print(f"Box Mean Precision:       {metrics["Box Precision"]:.3f}")
    print(f"Box Mean Recall:        {metrics["Box Recall"   ]:.3f}")
    print(f"Seg mAP@50:      {metrics["Mask mAP@50"  ]:.3f}")
    print(f"Seg mAP@50-95:   {metrics["Mask mAP@50-95"]:.3f}")
    print(f"Seg Mean F1:        {np.mean(np.array(metrics["Mask F1"])):.3f}")
    print(f"Seg Mean Precision:  {metrics["Mask Precision"]:.3f}")
    print(f"Seg Mean Recall: {metrics["Mask Recall"  ]:.3f}")

    # Visualize some predictions
    # visualize_predictions(model, test_loader, device)
    # visualize_predictions_with_ground_truth(model, test_loader, device)