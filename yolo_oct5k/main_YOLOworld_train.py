from ultralytics import YOLO, YOLOWorld

from constants import dataset_yaml_path, yoloworld_pretrained_model, yoloworld_model_name, yoloworld_model_path

if __name__ == '__main__':
    # If needed for frozen executables, uncomment the next line:
    # from multiprocessing import freeze_support; freeze_support()


    # Step 2. Initialize the YOLOv12 model.
    # Option 1: Start from scratch with config
    # model = YOLO(model_config)

    # Option 2: Start from pretrained weights (recommended)
    # Use a pre-trained YOLOv8 model (or YOLOv12 if available)

    model = YOLOWorld(yoloworld_pretrained_model)

    # Step 3. Train the model with enhanced parameters and augmentation
    results = model.train(
        data=dataset_yaml_path,  # dataset YAML config
        epochs=50,  # number of epochs
        imgsz=640,  # training image size
        batch=16,  # adjust according to your GPU memory
        optimizer="AdamW",  # try different optimizers
        lr0=0.0001,  # initial learning rate
        lrf=0.01,  # final learning rate as a fraction of initial lr
        weight_decay=0.0005,  # weight decay
        momentum=0.937,  # SGD momentum
        warmup_epochs=10.0,  # warmup epochs
        plots=True,
        cos_lr=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation settings
        hsv_h=0.015,  # hue augmentation
        hsv_s=0.7,  # saturation augmentation
        hsv_v=0.4,  # value augmentation (brightness)
        degrees=10.0,  # rotation (+/- deg)
        translate=0.1,  # translation (+/- fraction)
        scale=0.2,  # scale (+/- gain)
        fliplr=0.5,  # flip left-right probability
        mosaic=1.0,  # mosaic probability
        mixup=0.05,  # mixup probability
        close_mosaic=10,

        # Early stopping patience (if needed)
        patience=15,  # early stopping patience (epochs)

        # Save best model during training
        save_period=10,  # save checkpoint every x epochs
        project="runs",  # project name
        name=yoloworld_model_name,  # experiment name
    )

    print("Training completed.")
    # Save the current state of the model to a file.
    model.save(yoloworld_model_path)

    evaluation_results = model.val(
        data=dataset_yaml_path,
        imgsz=640,
        project="runs",
        name=yoloworld_model_name,
    )

    print(evaluation_results)

    # Evaluate the model on the validation set defined in data.yaml with different IoU thresholds
    # evaluation_results_lower_conf = model.val(
    #     data=dataset_yaml_path,
    #     imgsz=640,
    #     project="runs",
    #     name=model_name,
    #     conf=0.25,  # confidence threshold for evaluation
    #     iou=0.5,  # IoU threshold for NMS
    # )
    #
    # print(evaluation_results_lower_conf)
    #
    # # Optional: Evaluate with a different IoU threshold
    # evaluation_results_lower_iou = model.val(
    #     data=dataset_yaml_path,
    #     imgsz=640,
    #     project="runs",
    #     name=f"{model_name}_iou_0.4",
    #     conf=0.25,  # lower confidence threshold
    #     iou=0.4,  # lower IoU threshold
    # )
    #
    # print("Evaluation with lower IoU threshold:")
    # print(evaluation_results_lower_iou)