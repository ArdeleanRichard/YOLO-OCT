data_root = "../data/AROI/"
model_root = "./saved_models/"

# Step 1. Define your dataset YAML config.
dataset_yaml_path = "../" + data_root + 'saved_classes_all_sub_only_remap/data_aroi.yaml'
dataset_SEG_yaml_path = "../" + data_root + 'saved_classes_all_sub_only_remap_SEG/data_aroi.yaml'
suffix = ""

# dataset_yaml_path = "../" + data_root + 'saved_classes_all_sub_only_single_class/data_aroi.yaml'
# suffix = "_single_class"


# Instantiate the model.
yolo_pretrained_model = "yolo12s"
yolo_model_config = f"{yolo_pretrained_model}.pt"  # or use "yolov12.pt" if you have a pretrained model

yolo_model_name = yolo_pretrained_model + suffix
yolo_model_path = model_root + yolo_model_name + ".pt"





detr_pretrained_model = "rtdetr-l"
detr_model_config = f"{detr_pretrained_model}.pt"  # or use "yolov12.pt" if you have a pretrained model

detr_model_name = detr_pretrained_model + suffix
detr_model_path = model_root + detr_model_name + ".pt"





yoloe_pretrained_model = "yoloe-11s-seg"
yoloe_model_config = f"{yoloe_pretrained_model}.pt"  # or use "yolov12.pt" if you have a pretrained model

yoloe_model_name = yoloe_pretrained_model + suffix
yoloe_model_path = model_root + yoloe_model_name + ".pt"




yoloworld_pretrained_model = "yolov8s-worldv2"
yoloworld_model_config = f"{yoloworld_pretrained_model}.pt"  # or use "yolov12.pt" if you have a pretrained modelworld

yoloworld_model_name = yoloworld_pretrained_model + suffix
yoloworld_model_path = model_root + yoloworld_model_name + ".pt"