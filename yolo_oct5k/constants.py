data_root = "../data/OCT5k/"
model_root = "./saved_models/"

# Step 1. Define your dataset YAML config.
dataset_yaml_path = "../" + data_root + 'yolo/data_oct5k.yaml'
dataset_SEG_yaml_path = "../" + data_root + 'yolo_SEG/data_oct5k.yaml'
suffix = ""

# dataset_yaml_path = "../" + oct5k_root + 'yolo_balanced/data_oct5k_balanced.yaml'
# suffix = "_data_balanced"

# dataset_yaml_path = "../" + oct5k_root + 'yolo_enhanced/data_oct5k_enhanced.yaml'
# suffix = "_data_enhanced"

# dataset_yaml_path = "../" + oct5k_root + 'yolo_augmented/data_oct5k_augmented.yaml'
# suffix = "_data_augmented"

# dataset_yaml_path = "../" + oct5k_root + 'yolo_most_frequent/data_oct5k_most_frequent.yaml'
# suffix = "_most_frequent"

# dataset_yaml_path = "../" + oct5k_root + 'yolo_single_class/data_oct5k_single_class.yaml'
# suffix = "_single_class"

# dataset_yaml_path = "../" + oct5k_root + 'yolo_merged_data/data_oct5k.yaml'
# suffix = "_merged_data"


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