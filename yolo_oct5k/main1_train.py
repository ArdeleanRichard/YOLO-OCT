# https://docs.ultralytics.com/models/yolo12/#performance-metrics

from ultralytics import YOLO

from script_data_oct5k_converter_yolo import oct5k_root
from constants import model_config, model_path, model_name

if __name__ == '__main__':
    # If needed for frozen executables, uncomment the next line:
    # from multiprocessing import freeze_support; freeze_support()

    # Step 1. Define your dataset YAML config.
    # Adjust the paths below to point to your dataset directories.

    dataset_yaml_path = "../" + oct5k_root + 'yolo/data_oct5k.yaml'

    # Step 2. Initialize the YOLOv12 model.
    # For demonstration, we assume you use the model configuration file 'yolov12.yaml'
    # (which you would have from Ultralytics) or a pretrained 'yolov12.pt' file.
    #
    # If you want to start training from scratch you could use a configuration file,
    # or if pretrained weights are provided use that file.
    # model_config = "yolov12.yaml"
    # if not os.path.exists(model_config):
    #     print(f"Warning: {model_config} not found. Make sure you have the YOLOv12 model file.")


    model = YOLO(model_config)

    # Step 3. Train the model.
    # You can specify additional parameters like number of epochs, image size, etc.
    results = model.train(
        data=dataset_yaml_path,  # our dataset YAML config
        epochs=50,  # number of training epochs
        imgsz=640,  # training image size
        batch=16,  # batch size, adjust according to your GPU memory
    )

    print("Training completed.")
    # Save the current state of the model to a file.
    model.save(model_path)

    # Evaluate the model on the validation set defined in data.yaml
    evaluation_results = model.val(data=dataset_yaml_path, imgsz=640, project="runs", name=model_name)

    # The evaluation results typically include:
    # - Precision and recall
    # - mAP at different IOU thresholds (e.g., mAP@0.5, mAP@0.5:0.95)
    print(evaluation_results)


    # Optionally, you can perform validation or inference after training.
    # For example, to run inference on a single image:
    inference_results = model("../../data/OCT5k/yolo/images/test/AMD Part1_AMD (3).E2E_2- 25- 2017 9- 13- 55 PM_Image 16.png")
    print("Inference Results:")
    print(inference_results)