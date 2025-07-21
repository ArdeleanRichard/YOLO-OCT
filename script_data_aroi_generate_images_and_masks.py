import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import glob
from collections import Counter
import cv2

def organize_dataset(root_dir='AROI', output_dir='organized_data'):
    """
    Organizes the dataset by extracting files from all patient folders
    to create a structured dataset.

    Args:
        root_dir: Root directory of the original dataset
        output_dir: Directory to store the organized dataset
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Keep track of file pairs for easier visualization later
    file_pairs = []

    # Process each patient folder
    patient_dir = os.path.join(root_dir, '24 patient')
    for patient_folder in os.listdir(patient_dir):
        if not patient_folder.startswith('patient'):
            continue

        patient_path = os.path.join(patient_dir, patient_folder)

        # Source directories
        labeled_dir = os.path.join(patient_path, 'raw', 'labeled')
        mask_dir = os.path.join(patient_path, 'mask', 'number')

        # Check if directories exist
        if not os.path.exists(labeled_dir) or not os.path.exists(mask_dir):
            print(f"Warning: Missing directories for {patient_folder}")
            continue

        # Process image files
        for img_file in os.listdir(labeled_dir):
            if not (img_file.endswith('.png') or img_file.endswith('.jpg') or
                    img_file.endswith('.jpeg') or img_file.endswith('.tif')):
                continue

            # Construct new filenames with patient ID prefix for uniqueness
            new_img_name = f"{img_file}"
            src_img_path = os.path.join(labeled_dir, img_file)
            dst_img_path = os.path.join(images_dir, new_img_name)

            # Copy image file
            shutil.copy2(src_img_path, dst_img_path)

            # Look for corresponding mask file
            mask_file = img_file  # Assuming mask files have same name as images
            src_mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(src_mask_path):
                new_mask_name = f"{mask_file}"
                dst_mask_path = os.path.join(masks_dir, new_mask_name)
                shutil.copy2(src_mask_path, dst_mask_path)

                # Save file pair for visualization
                file_pairs.append((new_img_name, new_mask_name))

    print(f"Organized {len(file_pairs)} image-mask pairs")
    return file_pairs, images_dir, masks_dir


def organize_dataset_by_classes(root_dir='AROI', output_dir='filtered_data', target_classes=None):
    """
    Organizes the dataset by extracting files from all patient folders
    but only includes masks (and corresponding images) that contain at least one
    of the specified target classes.

    Args:
        root_dir: Root directory of the original dataset
        output_dir: Directory to store the filtered dataset
        target_classes: List of class numbers to filter by (e.g., [5, 6, 7])

    Returns:
        List of file pairs included in the filtered dataset
    """
    if target_classes is None:
        raise ValueError("Target classes must be specified")

    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Keep track of file pairs for easier visualization later
    file_pairs = []

    # Process each patient folder
    patient_dir = os.path.join(root_dir, '24 patient')
    for patient_folder in os.listdir(patient_dir):
        if not patient_folder.startswith('patient'):
            continue

        patient_path = os.path.join(patient_dir, patient_folder)

        # Source directories
        labeled_dir = os.path.join(patient_path, 'raw', 'labeled')
        mask_dir = os.path.join(patient_path, 'mask', 'number')

        # Check if directories exist
        if not os.path.exists(labeled_dir) or not os.path.exists(mask_dir):
            print(f"Warning: Missing directories for {patient_folder}")
            continue

        # Process mask files first to check for target classes
        for mask_file in os.listdir(mask_dir):
            if not (mask_file.endswith('.png') or mask_file.endswith('.jpg') or
                    mask_file.endswith('.jpeg') or mask_file.endswith('.tif')):
                continue

            # Check if mask contains any of the target classes
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = np.array(Image.open(mask_path))
                unique_values = np.unique(mask)

                # Check if any target class is present in the mask
                if not any(cls in unique_values for cls in target_classes):
                    continue  # Skip this mask if it doesn't contain any target class

                # Mask contains at least one target class, so include it and its corresponding image
                new_mask_name = f"{mask_file}"
                dst_mask_path = os.path.join(masks_dir, new_mask_name)
                shutil.copy2(mask_path, dst_mask_path)

                # Look for corresponding image file
                img_file = mask_file  # Assuming image files have same name as masks
                src_img_path = os.path.join(labeled_dir, img_file)

                if os.path.exists(src_img_path):
                    new_img_name = f"{img_file}"
                    dst_img_path = os.path.join(images_dir, new_img_name)
                    shutil.copy2(src_img_path, dst_img_path)

                    # Save file pair for visualization
                    file_pairs.append((new_img_name, new_mask_name))
                    # print(f"Included mask with classes {set(unique_values).intersection(target_classes)}: {new_mask_name}")
                else:
                    print(f"Warning: Found matching mask but no image for {mask_file} in {patient_folder}")

            except Exception as e:
                print(f"Error processing mask {mask_file}: {e}")

    print(f"Organized {len(file_pairs)} image-mask pairs containing target classes {target_classes}")
    return file_pairs, images_dir, masks_dir


def organize_dataset_by_classes_only(root_dir='AROI', output_dir='filtered_data', target_classes=None):
    """
    Organizes the dataset by extracting files from all patient folders
    but only includes masks (and corresponding images) that contain at least one
    of the specified target classes.

    Args:
        root_dir: Root directory of the original dataset
        output_dir: Directory to store the filtered dataset
        target_classes: List of class numbers to filter by (e.g., [5, 6, 7])

    Returns:
        List of file pairs included in the filtered dataset
    """
    if target_classes is None:
        raise ValueError("Target classes must be specified")

    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Keep track of file pairs for easier visualization later
    file_pairs = []

    # Process each patient folder
    patient_dir = os.path.join(root_dir, '24 patient')
    for patient_folder in os.listdir(patient_dir):
        if not patient_folder.startswith('patient'):
            continue

        patient_path = os.path.join(patient_dir, patient_folder)

        # Source directories
        labeled_dir = os.path.join(patient_path, 'raw', 'labeled')
        mask_dir = os.path.join(patient_path, 'mask', 'number')

        # Check if directories exist
        if not os.path.exists(labeled_dir) or not os.path.exists(mask_dir):
            print(f"Warning: Missing directories for {patient_folder}")
            continue

        # Process mask files first to check for target classes
        for mask_file in os.listdir(mask_dir):
            if not (mask_file.endswith('.png') or mask_file.endswith('.jpg') or
                    mask_file.endswith('.jpeg') or mask_file.endswith('.tif')):
                continue

            # Check if mask contains any of the target classes
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = np.array(Image.open(mask_path))
                unique_values = np.unique(mask)

                # Check if any target class is present in the mask
                if not any(cls in unique_values for cls in target_classes):
                    continue  # Skip this mask if it doesn't contain any target class

                # # Mask contains at least one target class, so include it and its corresponding image
                # new_mask_name = f"{mask_file}"
                # dst_mask_path = os.path.join(masks_dir, new_mask_name)
                # shutil.copy2(mask_path, dst_mask_path)

                # Modify mask to keep only target classes (set other classes to 0)
                modified_mask = np.zeros_like(mask)
                for cls in target_classes:
                    if cls in unique_values:
                        modified_mask[mask == cls] = cls

                # Create a new mask image with only target classes
                new_mask_name = f"{mask_file}"
                dst_mask_path = os.path.join(masks_dir, new_mask_name)
                Image.fromarray(modified_mask).save(dst_mask_path)

                # Look for corresponding image file
                img_file = mask_file  # Assuming image files have same name as masks
                src_img_path = os.path.join(labeled_dir, img_file)

                if os.path.exists(src_img_path):
                    new_img_name = f"{img_file}"
                    dst_img_path = os.path.join(images_dir, new_img_name)
                    shutil.copy2(src_img_path, dst_img_path)

                    # Save file pair for visualization
                    file_pairs.append((new_img_name, new_mask_name))
                    # print(f"Included mask with classes {set(unique_values).intersection(target_classes)}: {new_mask_name}")
                else:
                    print(f"Warning: Found matching mask but no image for {mask_file} in {patient_folder}")

            except Exception as e:
                print(f"Error processing mask {mask_file}: {e}")

    print(f"Organized {len(file_pairs)} image-mask pairs containing target classes {target_classes}")
    return file_pairs, images_dir, masks_dir


def organize_dataset_by_classes_only_remap(root_dir='AROI', output_dir='filtered_data', target_classes=None, class_mapping=None):
    """
    Organizes the dataset by extracting files from all patient folders
    but only includes masks (and corresponding images) that contain at least one
    of the specified target classes.

    Args:
        root_dir: Root directory of the original dataset
        output_dir: Directory to store the filtered dataset
        target_classes: List of class numbers to filter by (e.g., [5, 6, 7])

    Returns:
        List of file pairs included in the filtered dataset
    """
    if target_classes is None:
        raise ValueError("Target classes must be specified")

    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Keep track of file pairs for easier visualization later
    file_pairs = []

    # Process each patient folder
    patient_dir = os.path.join(root_dir, '24 patient')
    for patient_folder in os.listdir(patient_dir):
        if not patient_folder.startswith('patient'):
            continue

        patient_path = os.path.join(patient_dir, patient_folder)

        # Source directories
        labeled_dir = os.path.join(patient_path, 'raw', 'labeled')
        mask_dir = os.path.join(patient_path, 'mask', 'number')

        # Check if directories exist
        if not os.path.exists(labeled_dir) or not os.path.exists(mask_dir):
            print(f"Warning: Missing directories for {patient_folder}")
            continue

        # Process mask files first to check for target classes
        for mask_file in os.listdir(mask_dir):
            if not (mask_file.endswith('.png') or mask_file.endswith('.jpg') or
                    mask_file.endswith('.jpeg') or mask_file.endswith('.tif')):
                continue

            # Check if mask contains any of the target classes
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = np.array(Image.open(mask_path))
                unique_values = np.unique(mask)

                # Check if any target class is present in the mask
                if not any(cls in unique_values for cls in target_classes):
                    continue  # Skip this mask if it doesn't contain any target class

                # # Mask contains at least one target class, so include it and its corresponding image
                # new_mask_name = f"{mask_file}"
                # dst_mask_path = os.path.join(masks_dir, new_mask_name)
                # shutil.copy2(mask_path, dst_mask_path)

                # Modify mask to keep only target classes (set other classes to 0)
                modified_mask = np.zeros_like(mask)
                modified_mask = np.zeros_like(mask)
                for cls in target_classes:
                    if cls in unique_values:
                        if class_mapping and cls in class_mapping:
                            # Apply class mapping
                            modified_mask[mask == cls] = class_mapping[cls]
                        else:
                            # Keep original class value
                            modified_mask[mask == cls] = cls

                # Create a new mask image with only target classes
                new_mask_name = f"{mask_file}"
                dst_mask_path = os.path.join(masks_dir, new_mask_name)
                Image.fromarray(modified_mask).save(dst_mask_path)

                # Look for corresponding image file
                img_file = mask_file  # Assuming image files have same name as masks
                src_img_path = os.path.join(labeled_dir, img_file)

                if os.path.exists(src_img_path):
                    new_img_name = f"{img_file}"
                    dst_img_path = os.path.join(images_dir, new_img_name)
                    shutil.copy2(src_img_path, dst_img_path)

                    # Save file pair for visualization
                    file_pairs.append((new_img_name, new_mask_name))
                    # print(f"Included mask with classes {set(unique_values).intersection(target_classes)}: {new_mask_name}")
                else:
                    print(f"Warning: Found matching mask but no image for {mask_file} in {patient_folder}")

            except Exception as e:
                print(f"Error processing mask {mask_file}: {e}")

    print(f"Organized {len(file_pairs)} image-mask pairs containing target classes {target_classes}")
    return file_pairs, images_dir, masks_dir


def analyze_mask_classes(masks_dir):
    """
    Analyzes the distribution of classes in the mask images.

    Args:
        masks_dir: Directory containing mask images

    Returns:
        class_counts: Dictionary with counts of each class
    """
    class_counts = Counter()

    for mask_file in os.listdir(masks_dir):
        if not (mask_file.endswith('.png') or mask_file.endswith('.jpg') or
                mask_file.endswith('.jpeg') or mask_file.endswith('.tif')):
            continue

        try:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = np.array(Image.open(mask_path))

            # Count unique values in mask
            unique_values = np.unique(mask)
            for val in unique_values:
                if val > 0:  # Ignore background (0)
                    class_counts[int(val)] += 1
        except Exception as e:
            print(f"Error processing mask {mask_file}: {e}")

    return class_counts


def visualize_examples(file_pairs, images_dir, masks_dir, num_examples=5):
    """
    Visualizes random examples of images and their corresponding masks.

    Args:
        file_pairs: List of (image_filename, mask_filename) pairs
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        num_examples: Number of examples to visualize
    """
    if not file_pairs:
        print("No image-mask pairs found to visualize")
        return

    # Randomly select examples
    samples = random.sample(file_pairs, min(num_examples, len(file_pairs)))

    for i, (img_file, mask_file) in enumerate(samples):
        try:
            # Load image and mask
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, mask_file)

            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Display image
            ax1.imshow(img)
            ax1.set_title(f"Image: {img_file}")
            ax1.axis('off')

            # Display mask with colormap to distinguish classes
            mask_img = ax2.imshow(mask, cmap='tab10', vmin=0, vmax=9)
            ax2.set_title(f"Mask: {mask_file}")
            ax2.axis('off')

            # Add colorbar for class visualization
            unique_values = np.unique(mask)
            if len(unique_values) > 1:
                cbar = plt.colorbar(mask_img, ax=ax2, ticks=unique_values)
                cbar.set_label('Class')

            plt.tight_layout()
            plt.savefig(f"example_{i + 1}.png")
            plt.close()

            print(f"Saved visualization for example {i + 1}")
            print(f"Mask classes present: {np.unique(mask)}")

        except Exception as e:
            print(f"Error visualizing example {i + 1}: {e}")


def visualize_class_distribution(class_counts):
    """
    Visualizes the distribution of classes in the masks.

    Args:
        class_counts: Dictionary with counts of each class
    """
    if not class_counts:
        print("No class data to visualize")
        return

    plt.figure(figsize=(10, 6))
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]

    plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Masks Containing Class')
    plt.title('Distribution of Classes in Masks')
    plt.xticks(classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("class_distribution.png")
    plt.close()

    print("Saved class distribution visualization")


def main_create_data_full():
    root_dir = '../data/AROI/'
    output_dir = '../data/AROI/saved_classes_all/'

    file_pairs, images_dir, masks_dir = organize_dataset(root_dir, output_dir)

    print("Analyzing mask classes...")
    class_counts = analyze_mask_classes(masks_dir)
    print(f"Found the following classes: {sorted(class_counts.keys())}")
    print(f"Class distribution: {class_counts}")

    visualize_examples(file_pairs, images_dir, masks_dir, num_examples=5)
    visualize_class_distribution(class_counts)

def main_create_data_sub():
    root_dir = '../data/AROI/'
    output_dir = '../data/AROI/saved_classes_all_sub'

    file_pairs, images_dir, masks_dir = organize_dataset_by_classes(root_dir, output_dir, target_classes=[5,6,7])

    print("Analyzing mask classes...")
    class_counts = analyze_mask_classes(masks_dir)
    print(f"Found the following classes: {sorted(class_counts.keys())}")
    print(f"Class distribution: {class_counts}")

    visualize_examples(file_pairs, images_dir, masks_dir, num_examples=5)
    visualize_class_distribution(class_counts)


def main_create_data_sub_only():
    root_dir = '../data/AROI/'
    output_dir = '../data/AROI/saved_classes_all_sub_only'

    file_pairs, images_dir, masks_dir = organize_dataset_by_classes_only(root_dir, output_dir, target_classes=[5,6,7])

    print("Analyzing mask classes...")
    class_counts = analyze_mask_classes(masks_dir)
    print(f"Found the following classes: {sorted(class_counts.keys())}")
    print(f"Class distribution: {class_counts}")

    visualize_examples(file_pairs, images_dir, masks_dir, num_examples=5)
    visualize_class_distribution(class_counts)


def main_create_data_sub_only_remap(output_dir = '../data/AROI/saved_classes_all_sub_only_remap', target_classes=[5,6,7], class_mapping={5:1, 6:2, 7:3}):
    root_dir = '../data/AROI/'

    file_pairs, images_dir, masks_dir = organize_dataset_by_classes_only_remap(root_dir, output_dir, target_classes=target_classes, class_mapping=class_mapping)

    print("Analyzing mask classes...")
    class_counts = analyze_mask_classes(masks_dir)
    print(f"Found the following classes: {sorted(class_counts.keys())}")
    print(f"Class distribution: {class_counts}")

    visualize_examples(file_pairs, images_dir, masks_dir, num_examples=5)
    visualize_class_distribution(class_counts)


def visualize_mask_examples(images_dir, masks_dir, random_samples):
    """
    Visualizes random examples of images and their corresponding masks.

    Args:
        file_pairs: List of (image_filename, mask_filename) pairs
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        num_examples: Number of examples to visualize
    """



    for i, img_file in enumerate(random_samples):
        try:
            # Load image and mask
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)

            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Display image
            ax1.imshow(img)
            ax1.set_title(f"Image: {img_file}")
            ax1.axis('off')

            # Display mask with colormap to distinguish classes
            mask_img = ax2.imshow(mask, cmap='tab10', vmin=0, vmax=9)
            ax2.set_title(f"Mask: {img_file}")
            ax2.axis('off')

            # Add colorbar for class visualization
            unique_values = np.unique(mask)
            if len(unique_values) > 1:
                cbar = plt.colorbar(mask_img, ax=ax2, ticks=unique_values)
                cbar.set_label('Class')

            plt.tight_layout()
            plt.savefig(f"example_{i + 1}.png")
            plt.close()

            print(f"Saved visualization for example {i + 1}")
            print(f"Mask classes present: {np.unique(mask)}")

        except Exception as e:
            print(f"Error visualizing example {i + 1}: {e}")
def visualize_mask_examples(images_dir, masks_dir, random_samples):
    """
    Visualizes random examples of images and their corresponding masks.

    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        random_samples: List of image filenames to visualize
    """

    for i, img_file in enumerate(random_samples):
        try:
            # Load image and mask
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)

            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Create a figure with two subplots - consistent height
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Display image
            ax1.imshow(img)
            ax1.set_title(f"Image: {img_file}")
            ax1.axis('off')  # Remove axes

            # Display mask with colormap to distinguish classes
            mask_img = ax2.imshow(mask, cmap='tab10', vmin=0, vmax=9)
            ax2.set_title(f"Mask: {img_file}")
            ax2.axis('off')  # Remove axes

            # Add colorbar for class visualization
            unique_values = np.unique(mask)
            if len(unique_values) > 1:
                cbar = plt.colorbar(mask_img, ax=ax2, ticks=unique_values)
                cbar.set_label('Class')

            plt.tight_layout()
            plt.savefig(f"example_{i + 1}.png", bbox_inches='tight', dpi=150)
            plt.close()

            print(f"Saved visualization for example {i + 1}")
            print(f"Mask classes present: {np.unique(mask)}")

        except Exception as e:
            print(f"Error visualizing example {i + 1}: {e}")



def visualize_bbox_examples(images_dir, labels_dir, random_samples):
    """
    Visualizes random examples of images with their corresponding YOLO bounding boxes.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        random_samples: List of image filenames to visualize
    """

    for i, img_file in enumerate(random_samples):
        try:
            # Load image
            img_path = os.path.join(images_dir, img_file)
            img = np.array(Image.open(img_path))

            # Get base name for label lookup
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(labels_dir, base_name + '.txt')

            if not os.path.exists(label_file):
                print(f"No label file found for {img_file}")
                continue

            # Create figure for visualization - consistent height
            fig, ax = plt.subplots(1, figsize=(6, 6))  # Square format for consistency
            ax.imshow(img)
            ax.set_title(f"Image with YOLO bounding boxes: {img_file}")
            ax.axis('off')  # Remove axes

            # Get image dimensions
            height, width = img.shape[:2]

            # Read YOLO format labels
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Draw bounding boxes
            class_colors = {
                0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm',
                5: 'y', 6: 'w', 7: 'k', 8: 'orange', 9: 'purple'
            }

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id, x_center, y_center, w, h = map(float, parts)
                cls_id = int(cls_id)

                # Convert normalized coordinates back to pixel values
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                # Calculate box coordinates
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle (bounding box)
                color = class_colors.get(cls_id, 'r')  # Default to red if class not in colors
                rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # Add class label
                plt.text(x1, y1 - 5, f"Class {cls_id + 1}", color=color, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7))

            plt.tight_layout()
            plt.savefig(f"yolo_bbox_example_{i + 1}.png", bbox_inches='tight', dpi=150)
            plt.close()

            print(f"Saved visualization with bounding boxes for example {i + 1}")

        except Exception as e:
            print(f"Error visualizing bounding boxes for {img_file}: {e}")

    print("YOLO bounding box visualization complete")


def generate_yolo_labels(masks_dir, output_labels_dir):
    """
    Converts segmentation masks to YOLO format bounding boxes.

    Args:
        masks_dir: Directory containing mask images
        output_labels_dir: Directory to save YOLO format labels
    """
    os.makedirs(output_labels_dir, exist_ok=True)

    # Process each mask file
    for mask_file in os.listdir(masks_dir):
        if not (mask_file.endswith('.png') or mask_file.endswith('.jpg') or
                mask_file.endswith('.jpeg') or mask_file.endswith('.tif')):
            continue

        try:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = np.array(Image.open(mask_path))

            # Get image dimensions
            height, width = mask.shape

            # Create a text file for YOLO labels with the same base name as the mask
            base_name = os.path.splitext(mask_file)[0]
            label_file_path = os.path.join(output_labels_dir, base_name + '.txt')

            # Open file for writing YOLO labels
            with open(label_file_path, 'w') as f:
                # Get unique classes (excluding background class 0)
                unique_classes = np.unique(mask)
                unique_classes = unique_classes[unique_classes > 0]

                for cls in unique_classes:
                    # Create binary mask for this class
                    binary_mask = (mask == cls).astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Process each contour (each disconnected region of this class)
                    for contour in contours:
                        # Skip very small contours that might be noise
                        if cv2.contourArea(contour) < 10:  # Minimum area threshold
                            continue

                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Skip if width or height is zero
                        if w == 0 or h == 0:
                            continue

                        # Convert to YOLO format:
                        # class_id x_center y_center width height
                        # All values are normalized to [0, 1]
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        norm_width = w / width
                        norm_height = h / height

                        # Write to file in YOLO format
                        # Class is int(cls) - 1 since YOLO uses 0-indexed classes
                        # But we're using the actual class values here since they've already been remapped
                        f.write(f"{int(cls) - 1} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

            print(f"Generated YOLO labels for {mask_file}")

        except Exception as e:
            print(f"Error generating YOLO labels for {mask_file}: {e}")

    print(f"YOLO labels generation complete. Labels saved to {output_labels_dir}")
    return output_labels_dir


def main_generate_yolo_labels(output_dir='../data/AROI/saved_classes_all_sub_only_remap/'):
    images_dir = output_dir + "images/"
    masks_dir = output_dir + "masks/"
    labels_dir = os.path.join(output_dir, 'labels')
    labels_dir = generate_yolo_labels(masks_dir, labels_dir)

    # Visualize YOLO bounding boxes
    print("Visualizing YOLO bounding boxes...")

    num_examples=5
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    samples = random.sample(image_files, min(num_examples, len(image_files)))

    visualize_mask_examples(images_dir, masks_dir, random_samples=samples)
    visualize_bbox_examples(images_dir, labels_dir, random_samples=samples)


def create_train_val_test_split(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create train, validation, and test splits for a dataset with matching triplets of images, masks, and labels.

    Args:
        source_dir (str): Root directory containing 'images', 'masks', and 'labels' subfolders
        train_ratio (float): Ratio of data for training set (default: 0.7)
        val_ratio (float): Ratio of data for validation set (default: 0.15)
        test_ratio (float): Ratio of data for test set (default: 0.15)
        seed (int): Random seed for reproducibility
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Set random seed for reproducibility
    random.seed(seed)

    # Get paths
    source_dir = Path(source_dir)
    images_dir = source_dir / 'images'
    masks_dir = source_dir / 'masks'
    labels_dir = source_dir / 'labels'

    # Verify all directories exist
    for directory in [images_dir, masks_dir, labels_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} not found")

    # Get all image filenames
    image_files = [f for f in os.listdir(images_dir) if not f.startswith('.')]

    # Check if there are matching files in masks and labels directories
    for img_file in image_files:
        # Check for mask with the same name
        if not (masks_dir / img_file).exists():
            print(f"Warning: No matching mask file for {img_file}")

        # For label files, check for corresponding .txt file (YOLO format)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if not (labels_dir / label_file).exists():
            print(f"Warning: No matching label file for {img_file} (looked for {label_file})")

    # Shuffle the file list
    random.shuffle(image_files)

    # Calculate split sizes
    total_files = len(image_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # Split the files
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    # Print split stats
    print(f"Total files: {total_files}")
    print(f"Train files: {len(train_files)} ({len(train_files) / total_files:.2%})")
    print(f"Val files: {len(val_files)} ({len(val_files) / total_files:.2%})")
    print(f"Test files: {len(test_files)} ({len(test_files) / total_files:.2%})")

    # Create destination directories
    splits = ['train', 'val', 'test']
    categories = ['images', 'masks', 'labels']

    for split in splits:
        for category in categories:
            os.makedirs(source_dir / category / split, exist_ok=True)

    # Copy files according to splits
    def copy_files(file_list, split_name):
        for img_filename in file_list:
            # Copy image to its split directory
            if (images_dir / img_filename).exists():
                shutil.copy(images_dir / img_filename, images_dir / split_name / img_filename)

            # Copy mask to its split directory (same filename as image)
            if (masks_dir / img_filename).exists():
                shutil.copy(masks_dir / img_filename, masks_dir / split_name / img_filename)

            # Copy label (txt file) to its split directory
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            if (labels_dir / label_filename).exists():
                shutil.copy(labels_dir / label_filename, labels_dir / split_name / label_filename)

    # Perform the copying
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    print("Dataset split complete!")


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


# Usage
if __name__ == "__main__":
    # main_create_data_full()
    # main_create_data_sub()
    # main_create_data_sub_only()
    # main_create_data_sub_only_remap()
    # main_generate_yolo_labels()

    # output_dir = '../data/AROI/saved_classes_all_sub_only_remap3/'
    # main_create_data_sub_only_remap(output_dir, target_classes=[5, 6, 7], class_mapping={5: 1, 6: 2, 7: 3})
    # main_generate_yolo_labels(output_dir)
    # create_train_val_test_split(output_dir, 0.7, 0.2, 0.1)

    # OCT 5k prep
    # output_dir = '../data/AROI/oct5k_merge/'
    # main_create_data_sub_only_remap(output_dir, target_classes=[5, 6, 7], class_mapping={5: 1, 6: 1, 7: 8})
    # main_generate_yolo_labels(output_dir)
    # create_train_val_test_split(output_dir, 0.7, 0.2, 0.1)

    # output_dir = '../data/AROI/saved_classes_all_sub_only_single_class/'
    # main_create_data_sub_only_remap(output_dir, target_classes=[5, 6, 7], class_mapping={5: 1, 6: 1, 7: 1})
    # main_generate_yolo_labels(output_dir)
    # create_train_val_test_split(output_dir, 0.7, 0.2, 0.1)

    convert_yolo_to_seg("../data/AROI/saved_classes_all_sub_only_remap/", "../data/AROI/saved_classes_all_sub_only_remap_SEG/")

