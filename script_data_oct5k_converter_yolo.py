import glob
import os
import random
from shutil import move, copy2

import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

oct5k_root = "../data/OCT5k/"
oct5k_folder_detections= oct5k_root + "Detection/"

file_bboxs = "all_bounding_boxes.csv"
file_classes = "all_classes.csv"


# Folders
IMG_DIR =           oct5k_root + 'Images/'   # Directory where your image files live
YOLO_IMG_DIR =      oct5k_root + 'yolo/images/'          # Output directory for copied/organized images (optional)
YOLO_LABEL_DIR =    oct5k_root + 'yolo/labels/'        # Output directory for YOLO-format .txt files
YOLO_MASKS_DIR =    oct5k_root + 'yolo/masks/'        # Output directory for YOLO-format .txt files

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

CONFIG_FILE = oct5k_root + 'yolo/data_oct5k.yaml'    # YOLO dataset config output

def create_dataset_for_yolo():
    # Create output directories if they don't exist
    os.makedirs(YOLO_LABEL_DIR, exist_ok=True)
    os.makedirs(YOLO_IMG_DIR, exist_ok=True)

    # --- LOAD DATA ---
    bboxes = pd.read_csv(oct5k_folder_detections + file_bboxs)
    class_map_df = pd.read_csv(oct5k_folder_detections + file_classes, header=None, names=['class', 'index'])



    # Read the class map CSV without header
    # It has two columns: class_name, class_index
    class_map = dict(zip(class_map_df['class'], class_map_df['index']))

    # Process each image group
    for img_rel, group in bboxes.groupby('image'):
        # Locate the image under IMG_ROOT
        pattern = os.path.join(IMG_DIR, '**', img_rel)
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not find image: {img_rel}")
        img_path = matches[0]

        # Read dimensions
        with Image.open(img_path) as img:
            w, h = img.size

        # Create a flat, unique basename by sanitizing the relative path
        # e.g. 'AMD Part1/AMD (3)/Image (14).png' -> 'AMD Part1_AMD (3)_Image (14).png'
        sanitized = img_rel.replace('/', '_').replace('\\', '_')
        name_no_ext, ext = os.path.splitext(sanitized)

        # Prepare YOLO label file
        label_file = os.path.join(YOLO_LABEL_DIR, name_no_ext + '.txt')
        yolo_lines = []
        for _, row in group.iterrows():
            cls_name = row['class']
            if cls_name not in class_map:
                raise KeyError(f"Class '{cls_name}' not in map.")
            cls_idx = class_map[cls_name]
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            x_c = (xmin + xmax) / 2.0 / w
            y_c = (ymin + ymax) / 2.0 / h
            bb_w = (xmax - xmin) / w
            bb_h = (ymax - ymin) / h
            yolo_lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {bb_w:.6f} {bb_h:.6f}")
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))

        # Copy or link image into flat YOLO_IMG_DIR
        dest_img = os.path.join(YOLO_IMG_DIR, sanitized)
        if not os.path.exists(dest_img):
            from shutil import copy2
            copy2(img_path, dest_img)

    print("Done! All labels in 'labels/' and images in 'yolo_images/' (flat structure).")


def split_into_subfolders():
    def ensure_dirs(base_dir):
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(base_dir, split), exist_ok=True)

    ensure_dirs(YOLO_IMG_DIR)
    ensure_dirs(YOLO_LABEL_DIR)

    # --- SPLIT INTO TRAIN/VAL/TEST ---
    print('Splitting data into train/val/test...')
    all_images = [f for f in os.listdir(YOLO_IMG_DIR) if os.path.isfile(os.path.join(YOLO_IMG_DIR, f))]
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for split, files in splits.items():
        for fname in files:
            # Move image
            src_img = os.path.join(YOLO_IMG_DIR, fname)
            dst_img = os.path.join(YOLO_IMG_DIR, split, fname)
            move(src_img, dst_img)
            # Move corresponding label
            label_name = os.path.splitext(fname)[0] + '.txt'
            src_lbl = os.path.join(YOLO_LABEL_DIR, label_name)
            dst_lbl = os.path.join(YOLO_LABEL_DIR, split, label_name)
            if os.path.exists(src_lbl):
                move(src_lbl, dst_lbl)
            else:
                print(f"Warning: label file not found for {fname}")

    print('Dataset split complete!')


def check():
    bboxes = pd.read_csv(oct5k_folder_detections + file_bboxs)
    unique_names = bboxes['image'].nunique()
    print("Number of unique images:", unique_names)


def write_config_yaml():
    # --- WRITE YOLO CONFIG FILE ---
    class_map_df = pd.read_csv(oct5k_folder_detections + file_classes, header=None, names=['class', 'index'])

    num_classes = len(class_map_df)
    class_names = class_map_df['class'].tolist()
    print(class_names)
    config = {
        'train': os.path.abspath(os.path.join(YOLO_IMG_DIR, 'train')),
        'val': os.path.abspath(os.path.join(YOLO_IMG_DIR, 'val')),
        'nc': num_classes,
        'names': class_names # [{', '.join(f"{n}" for n in class_names)}]
    }
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, width=float("inf"))
    print(f"Wrote YOLO dataset config to {CONFIG_FILE}")


def show_example_yolo():
    import os
    import cv2
    import matplotlib.pyplot as plt

    # Choose your image and label file
    image_path = oct5k_root + f'yolo/images/train/AMD Part1_AMD (1).E2E_2- 25- 2017 9- 10- 42 PM_Image 14.png'
    label_path = oct5k_root + f'yolo/labels/train/AMD Part1_AMD (1).E2E_2- 25- 2017 9- 10- 42 PM_Image 14.txt'

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Read label file
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Plot image and boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines

        class_id, x_center, y_center, box_w, box_h = map(float, parts)
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x_min = int(x_center - box_w / 2)
        y_min = int(y_center - box_h / 2)

        rect = plt.Rectangle((x_min, y_min), box_w, box_h,
                             linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x_min, y_min - 5, f'Class {int(class_id)}', color='red',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))

    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.savefig("test_yolo.png")
    plt.close()


def create_segmentation_masks(yolo_dir=YOLO_IMG_DIR):
    from PIL import Image, ImageDraw

    os.makedirs(YOLO_MASKS_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        img_split_dir = os.path.join(YOLO_IMG_DIR, split)
        label_split_dir = os.path.join(YOLO_LABEL_DIR, split)
        mask_split_dir = os.path.join(YOLO_MASKS_DIR, split)
        os.makedirs(mask_split_dir, exist_ok=True)

        for img_name in os.listdir(img_split_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(img_split_dir, img_name)
            label_path = os.path.join(label_split_dir, os.path.splitext(img_name)[0] + '.txt')

            if not os.path.exists(label_path):
                print(f"No label found for {img_name}, skipping.")
                continue

            # Load image size
            with Image.open(img_path) as img:
                w, h = img.size

            # Create blank mask (black)
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)

            # Parse YOLO label and draw rectangles
            # Multi-class mask (start all pixels as background class: 0)
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_idx, x_c, y_c, box_w, box_h = map(float, parts)
                    cls_idx = int(cls_idx)

                    x_center = x_c * w
                    y_center = y_c * h
                    box_width = box_w * w
                    box_height = box_h * h

                    xmin = int(x_center - box_width / 2)
                    ymin = int(y_center - box_height / 2)
                    xmax = int(x_center + box_width / 2)
                    ymax = int(y_center + box_height / 2)

                    mask[ymin:ymax, xmin:xmax] = cls_idx + 1  # Add 1 to separate from background (0)

            mask_path = os.path.join(mask_split_dir, f"{img_name}")
            Image.fromarray(mask).save(mask_path)

    print("Segmentation masks created.")


def show_example_mask(alpha=0.5):
    """
    Display an image with its segmentation mask overlaid.

    Parameters:
    - image_path: Path to the original image
    - mask_path: Path to the corresponding mask image
    - alpha: Transparency level of the mask (0 = invisible, 1 = fully opaque)
    """
    image_path = oct5k_root + f'yolo/images/train/AMD Part1_AMD (1).E2E_2- 25- 2017 9- 10- 42 PM_Image 14.png'
    mask_path = oct5k_root + f'yolo/masks/train/AMD Part1_AMD (1).E2E_2- 25- 2017 9- 10- 42 PM_Image 14.png'

    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Grayscale

    # Convert to numpy arrays
    img_np = np.array(img)
    mask_np = np.array(mask)

    # Define a color map (skip background class 0)
    num_classes = mask_np.max()
    cmap = plt.get_cmap('tab10', num_classes + 1)  # or use 'Set3', 'nipy_spectral', etc.

    # Create a color overlay for the mask
    color_mask = cmap(mask_np)
    color_mask[..., 3] = (mask_np > 0) * alpha  # set alpha only for non-background

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.imshow(color_mask)
    plt.axis('off')
    plt.title("Image with Multi-class Segmentation Mask")

    class_map_df = pd.read_csv(oct5k_folder_detections + file_classes, header=None, names=['class', 'index'])
    class_names = class_map_df['class'].tolist()

    # Optional: show legend
    if class_names:
        from matplotlib.patches import Patch
        handles = [Patch(color=cmap(i), label=class_names[i - 1]) for i in range(1, num_classes + 1)]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("test_mask")
    plt.close()


def shows_examples_masks(
    image_dir,
    mask_dir,
    output_dir,
    class_map_csv,
    alpha=0.5
):
    """
    Overlay segmentation masks on images in a directory and save the results.

    Parameters:
    - image_dir: Directory containing original images
    - mask_dir: Directory containing corresponding masks (same filenames)
    - output_dir: Directory to save the output images with masks
    - class_map_csv: Path to CSV file mapping class names to indices
    - alpha: Transparency level of the mask (0 = invisible, 1 = fully opaque)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load class names
    class_map_df = pd.read_csv(class_map_csv, header=None, names=['class', 'index'])
    class_names = class_map_df['class'].tolist()

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping.")
            continue

        # Load image and mask
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img_np = np.array(img)
        mask_np = np.array(mask)

        num_classes = mask_np.max()
        cmap = plt.get_cmap('tab10', num_classes + 1)

        color_mask = cmap(mask_np)
        color_mask[..., 3] = (mask_np > 0) * alpha

        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.imshow(color_mask)
        plt.axis('off')
        plt.title(f"Image: {filename}")

        # Optional legend
        if class_names:
            handles = [Patch(color=cmap(i), label=class_names[i - 1]) for i in range(1, num_classes + 1)]
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask_overlay.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Processed and saved overlays for all images in: {output_dir}")

def create_dataset_most_frequent_class():
    """
    Creates a dataset containing only images and bboxes for the most frequent class.
    Copies relevant images and creates new label files with only the most frequent class.
    """
    # Create output directories if they don't exist
    YOLO_IMG_DIR_MOST_FREQ = oct5k_root + 'yolo_most_frequent/images/'
    YOLO_LABEL_DIR_MOST_FREQ = oct5k_root + 'yolo_most_frequent/labels/'

    os.makedirs(YOLO_IMG_DIR_MOST_FREQ, exist_ok=True)
    os.makedirs(YOLO_LABEL_DIR_MOST_FREQ, exist_ok=True)

    # Load bounding box data and class mapping
    bboxes = pd.read_csv(oct5k_folder_detections + file_bboxs)
    class_map_df = pd.read_csv(oct5k_folder_detections + file_classes, header=None, names=['class', 'index'])
    class_map = dict(zip(class_map_df['class'], class_map_df['index']))

    # Find the most frequent class
    class_counts = bboxes['class'].value_counts()
    most_frequent_class = class_counts.idxmax()
    most_frequent_class_idx = class_map[most_frequent_class]

    print(f"Most frequent class: {most_frequent_class} (index: {most_frequent_class_idx})")
    print(f"Count: {class_counts[most_frequent_class]} out of {len(bboxes)} bboxes")

    # Filter bboxes for the most frequent class
    filtered_bboxes = bboxes[bboxes['class'] == most_frequent_class]

    # Get unique images that contain the most frequent class
    unique_images = filtered_bboxes['image'].unique()
    print(f"Found {len(unique_images)} images containing class '{most_frequent_class}'")

    # Process each image
    processed_count = 0
    for img_rel in unique_images:
        # Locate the image under IMG_ROOT
        pattern = os.path.join(IMG_DIR, '**', img_rel)
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"Could not find image: {img_rel}, skipping.")
            continue

        img_path = matches[0]

        # Read dimensions
        with Image.open(img_path) as img:
            w, h = img.size

        # Create a flat, unique basename by sanitizing the relative path
        sanitized = img_rel.replace('/', '_').replace('\\', '_')
        name_no_ext, ext = os.path.splitext(sanitized)

        # Get bboxes for this image that match our class
        img_bboxes = filtered_bboxes[filtered_bboxes['image'] == img_rel]

        if len(img_bboxes) == 0:
            continue  # Skip if no matching bboxes (shouldn't happen with our filtering)

        # Prepare YOLO label file
        label_file = os.path.join(YOLO_LABEL_DIR_MOST_FREQ, name_no_ext + '.txt')
        yolo_lines = []

        for _, row in img_bboxes.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            x_c = (xmin + xmax) / 2.0 / w
            y_c = (ymin + ymax) / 2.0 / h
            bb_w = (xmax - xmin) / w
            bb_h = (ymax - ymin) / h
            # Use class index 0 for single-class dataset
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {bb_w:.6f} {bb_h:.6f}")

        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))

        # Copy image to output directory
        dest_img = os.path.join(YOLO_IMG_DIR_MOST_FREQ, sanitized)
        if not os.path.exists(dest_img):
            copy2(img_path, dest_img)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images")

    # Create splits directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_IMG_DIR_MOST_FREQ, split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_LABEL_DIR_MOST_FREQ, split), exist_ok=True)

    # Split data into train/val/test
    all_images = [f for f in os.listdir(YOLO_IMG_DIR_MOST_FREQ)
                  if os.path.isfile(os.path.join(YOLO_IMG_DIR_MOST_FREQ, f))]
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for split, files in splits.items():
        for fname in files:
            # Move image
            src_img = os.path.join(YOLO_IMG_DIR_MOST_FREQ, fname)
            dst_img = os.path.join(YOLO_IMG_DIR_MOST_FREQ, split, fname)
            move(src_img, dst_img)

            # Move corresponding label
            label_name = os.path.splitext(fname)[0] + '.txt'
            src_lbl = os.path.join(YOLO_LABEL_DIR_MOST_FREQ, label_name)
            dst_lbl = os.path.join(YOLO_LABEL_DIR_MOST_FREQ, split, label_name)

            if os.path.exists(src_lbl):
                move(src_lbl, dst_lbl)
            else:
                print(f"Warning: label file not found for {fname}")

    # Write YAML config for single-class dataset
    config_file = oct5k_root + 'yolo_most_frequent/data.yaml'
    config = {
        'train': os.path.abspath(os.path.join(YOLO_IMG_DIR_MOST_FREQ, 'train')),
        'val': os.path.abspath(os.path.join(YOLO_IMG_DIR_MOST_FREQ, 'val')),
        'nc': 1,  # Only one class
        'names': [most_frequent_class]  # Name of the most frequent class
    }

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, width=float("inf"))

    print(f"Created single-class dataset with most frequent class: {most_frequent_class}")
    print(f"Wrote config to {config_file}")
    print(f"Total images: {n} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")


def create_dataset_single_class():
    """
    Creates a dataset with all bboxes from all classes but maps them all to a single class (class 0).
    This effectively creates a binary detection dataset (object vs background).
    """
    # Create output directories
    YOLO_IMG_DIR_SINGLE = oct5k_root + 'yolo_single_class/images/'
    YOLO_LABEL_DIR_SINGLE = oct5k_root + 'yolo_single_class/labels/'

    os.makedirs(YOLO_IMG_DIR_SINGLE, exist_ok=True)
    os.makedirs(YOLO_LABEL_DIR_SINGLE, exist_ok=True)

    # Load bounding box data
    bboxes = pd.read_csv(oct5k_folder_detections + file_bboxs)

    # Process each image group
    print("Processing images for single-class dataset...")
    for img_rel, group in bboxes.groupby('image'):
        # Locate the image under IMG_ROOT
        pattern = os.path.join(IMG_DIR, '**', img_rel)
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"Could not find image: {img_rel}, skipping.")
            continue

        img_path = matches[0]

        # Read dimensions
        with Image.open(img_path) as img:
            w, h = img.size

        # Create a flat, unique basename
        sanitized = img_rel.replace('/', '_').replace('\\', '_')
        name_no_ext, ext = os.path.splitext(sanitized)

        # Prepare YOLO label file with all bboxes set to class 0
        label_file = os.path.join(YOLO_LABEL_DIR_SINGLE, name_no_ext + '.txt')
        yolo_lines = []

        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            x_c = (xmin + xmax) / 2.0 / w
            y_c = (ymin + ymax) / 2.0 / h
            bb_w = (xmax - xmin) / w
            bb_h = (ymax - ymin) / h
            # Always use class 0
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {bb_w:.6f} {bb_h:.6f}")

        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))

        # Copy image to output directory
        dest_img = os.path.join(YOLO_IMG_DIR_SINGLE, sanitized)
        if not os.path.exists(dest_img):
            copy2(img_path, dest_img)

    # Create split directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_IMG_DIR_SINGLE, split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_LABEL_DIR_SINGLE, split), exist_ok=True)

    # Split data
    all_images = [f for f in os.listdir(YOLO_IMG_DIR_SINGLE)
                  if os.path.isfile(os.path.join(YOLO_IMG_DIR_SINGLE, f))]
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    print("Splitting data into train/val/test...")
    for split, files in splits.items():
        for fname in files:
            # Move image
            src_img = os.path.join(YOLO_IMG_DIR_SINGLE, fname)
            dst_img = os.path.join(YOLO_IMG_DIR_SINGLE, split, fname)
            move(src_img, dst_img)

            # Move corresponding label
            label_name = os.path.splitext(fname)[0] + '.txt'
            src_lbl = os.path.join(YOLO_LABEL_DIR_SINGLE, label_name)
            dst_lbl = os.path.join(YOLO_LABEL_DIR_SINGLE, split, label_name)

            if os.path.exists(src_lbl):
                move(src_lbl, dst_lbl)
            else:
                print(f"Warning: label file not found for {fname}")

    # Write YAML config
    config_file = oct5k_root + 'yolo_single_class/data.yaml'
    config = {
        'train': os.path.abspath(os.path.join(YOLO_IMG_DIR_SINGLE, 'train')),
        'val': os.path.abspath(os.path.join(YOLO_IMG_DIR_SINGLE, 'val')),
        'nc': 1,  # Single class
        'names': ['object']  # Generic object name
    }

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, width=float("inf"))

    print(f"Created single-class dataset (all objects mapped to class 0)")
    print(f"Wrote config to {config_file}")
    print(f"Total images: {n} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")



import os
import shutil
from pathlib import Path


def convert_yolo_to_seg(input_folder, output_folder):
    """
    Convert YOLO detection dataset to segmentation format

    Args:
        input_folder: Path to existing YOLO dataset
        output_folder: Path where segmentation dataset will be created
    """
    import shutil
    import yaml
    from pathlib import Path

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output structure
    output_path.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        input_images = input_path / 'images' / split
        input_labels = input_path / 'labels' / split

        if not input_images.exists():
            print(f"Warning: {input_images} does not exist, skipping {split}")
            continue

        # Copy images
        for img_file in input_images.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                shutil.copy2(img_file, output_path / 'images' / split / img_file.name)

        # Convert labels
        for label_file in input_labels.glob('*.txt'):
            new_labels = []

            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Skip empty files
            if not lines:
                continue

            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_id = parts[0]
                        x_center, y_center, width, height = map(float, parts[1:5])

                        # Validate bbox coordinates
                        if width <= 0 or height <= 0:
                            print(f"Warning: Invalid bbox dimensions in {label_file.name}: w={width}, h={height}")
                            continue

                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            print(f"Warning: Invalid bbox center in {label_file.name}: x={x_center}, y={y_center}")
                            continue

                        # Calculate bbox corners with proper bounds checking
                        x1 = max(0.0, x_center - width / 2)
                        y1 = max(0.0, y_center - height / 2)
                        x2 = min(1.0, x_center + width / 2)
                        y2 = min(1.0, y_center + height / 2)

                        # Ensure we have a valid rectangle
                        if x2 <= x1 or y2 <= y1:
                            print(f"Warning: Degenerate bbox in {label_file.name}: ({x1},{y1}) to ({x2},{y2})")
                            continue

                        # Create rectangle polygon in correct order (counterclockwise)
                        # Format: class_id x1 y1 x2 y1 x2 y2 x1 y2
                        polygon_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

                        # Format as string with proper precision
                        coord_str = ' '.join([f"{coord:.6f}" for coord in polygon_coords])
                        polygon = f"{class_id} {coord_str}"
                        new_labels.append(polygon)

                    except ValueError as e:
                        print(f"Warning: Could not parse line in {label_file.name}: {line} - {e}")
                        continue

            # Only write file if we have valid labels
            if new_labels:
                output_label_path = output_path / 'labels' / split / label_file.name
                with open(output_label_path, 'w') as f:
                    for label in new_labels:
                        f.write(label + '\n')
            else:
                print(f"Warning: No valid labels found in {label_file.name}")


    print(f"Conversion complete: {output_folder}")
    print("Dataset structure:")
    print(f"  Images: {len(list((output_path / 'images' / 'train').glob('*')))} train, {len(list((output_path / 'images' / 'val').glob('*')))} val, {len(list((output_path / 'images' / 'test').glob('*')))} test")
    print(f"  Labels: {len(list((output_path / 'labels' / 'train').glob('*')))} train, {len(list((output_path / 'labels' / 'val').glob('*')))} val, {len(list((output_path / 'labels' / 'test').glob('*')))} test")


if __name__ == "__main__":
    # create_dataset_for_yolo()
    # split_into_subfolders()
    # write_config_yaml()
    # check()
    # create_segmentation_masks()
    # show_example_yolo()
    # show_example_mask()

    shows_examples_masks(
        image_dir=oct5k_root + 'yolo/images/train',
        mask_dir=oct5k_root + 'yolo/masks/train',
        output_dir='./oct5k_examples',
        class_map_csv=oct5k_folder_detections + file_classes,
        alpha=0.5
    )

    # create_dataset_most_frequent_class()
    # create_dataset_single_class()

    # convert_yolo_to_seg("../data/OCT5k/yolo/", "../data/OCT5k/yolo_SEG/")