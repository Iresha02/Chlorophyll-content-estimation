import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def get_all_image_paths(root_dir, extensions=(".jpg", ".png", ".jpeg")):
    image_dict = defaultdict(list)
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.suffix.lower() in extensions:
            parent_folder = path.parts[-2]  # immediate parent folder
            image_dict[parent_folder].append(str(path))
    
    # Sort image paths to ensure 000nm comes first
    for folder in image_dict:
        image_dict[folder] = sorted(image_dict[folder])
    
    return image_dict

def load_dark_image(dataset_root):
    """
    Load the dark current image from the dataset.
    Dark image is named with '000nm' in the filename.
    """
    root = Path(dataset_root)
    dark_image_path = None
    
    # Search for dark image with '000nm' in filename
    for path in root.rglob("*"):
        if path.is_file() and '000nm' in path.stem.lower():
            dark_image_path = str(path)
            break
    
    if dark_image_path:
        dark_img = cv2.imread(dark_image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Dark image loaded from: {dark_image_path}")
        print(f"Dark image shape: {dark_img.shape}")
        return dark_img
    else:
        print("Warning: No dark image (000nm) found. Proceeding without dark current reduction.")
        return None

def subtract_dark_current(image, dark_image):
    """
    Subtract dark current from the image.
    """
    if dark_image is None:
        return image
    
    # Ensure both images are the same size
    if image.shape != dark_image.shape:
        print(f"Warning: Resizing dark image from {dark_image.shape} to {image.shape}")
        dark_image = cv2.resize(dark_image, (image.shape[1], image.shape[0]))
    
    # Convert to float for subtraction to avoid clipping
    img_float = image.astype(np.float32)
    dark_float = dark_image.astype(np.float32)
    
    # Subtract and clip to valid range [0, 255]
    corrected = np.clip(img_float - dark_float, 0, 255)
    
    return corrected.astype(np.uint8)

def is_dark_image(image_path):
    """
    Check if the image is the dark current image (000nm).
    """
    return '000nm' in Path(image_path).stem.lower()

def show_sample_image_and_get_crop_point(image_path, dark_image=None):
    """
    Display sample image (after dark current reduction) and get crop point.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply dark current reduction if available
    if dark_image is not None:
        img = subtract_dark_current(img, dark_image)
    
    crop_point = []
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            crop_point.append((int(event.xdata), int(event.ydata)))
            # Draw a crosshair at the clicked point
            ax.plot(event.xdata, event.ydata, 'r+', markersize=20, markeredgewidth=2)
            # Draw the crop box
            w, h = 100, 100
            rect = plt.Rectangle((event.xdata - w/2, event.ydata - h/2), 
                                w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            plt.draw()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title("Click the center point for 100x100 crop\n(Image shown after dark current reduction)")
    ax.grid(True, alpha=0.3)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return crop_point[0] if crop_point else None

def crop_and_save_all_images(image_dict, crop_center, save_root, dark_image=None, 
                             crop_size=(100, 100), original_root=None):
    """
    Apply dark current reduction, then crop and save all images.
    Skips the dark image (000nm) itself.
    """
    cx, cy = crop_center
    w, h = crop_size
    save_root = Path(save_root)
    original_root = Path(original_root)
    
    processed_count = 0
    skipped_count = 0
    
    for folder, image_paths in image_dict.items():
        print(f"\nProcessing folder: {folder}")
        
        for img_path in image_paths:
            # Skip dark image (000nm)
            if is_dark_image(img_path):
                print(f"  Skipping dark image: {Path(img_path).name}")
                skipped_count += 1
                continue
            
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"  Warning: Could not load {img_path}")
                skipped_count += 1
                continue
            
            # Step 1: Dark Current Reduction
            if dark_image is not None:
                img = subtract_dark_current(img, dark_image)
            
            # Step 2: Cropping
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = x1 + w
            y2 = y1 + h
            
            # Ensure crop doesn't exceed image boundaries
            if y2 > img.shape[0]:
                y2 = img.shape[0]
                y1 = y2 - h
            if x2 > img.shape[1]:
                x2 = img.shape[1]
                x1 = x2 - w
            
            crop = img[y1:y2, x1:x2]
            
            # Verify crop size
            if crop.shape[0] != h or crop.shape[1] != w:
                print(f"  Warning: Crop size mismatch for {Path(img_path).name}: {crop.shape}")
            
            # Construct save path
            img_path_obj = Path(img_path)
            relative_path = img_path_obj.relative_to(original_root)
            save_path = save_root / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cropped image
            success = cv2.imwrite(str(save_path), crop)
            if success:
                processed_count += 1
                if processed_count % 10 == 0:  # Progress indicator
                    print(f"  Processed {processed_count} images...")
            else:
                print(f"  Error: Failed to save {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images (including dark images)")
    print(f"{'='*60}")

def main():
    # Dataset root folder
    dataset_root = r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\Chlorophyll MSI"
    
    save_root = Path(dataset_root).parent / f"Preprocessed_{Path(dataset_root).name}"
    
    print("="*60)
    print("MULTISPECTRAL IMAGE PREPROCESSING")
    print("Step 1: Dark Current Reduction")
    print("Step 2: Image Cropping")
    print("="*60)
    
    print("\n" + "="*60)
    print("STEP 1: Loading Dark Image (000nm)")
    print("="*60)
    
    # Load dark image
    dark_image = load_dark_image(dataset_root)
    
    if dark_image is None:
        response = input("\nNo dark image found. Continue without dark current reduction? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    print("\n" + "="*60)
    print("STEP 2: Loading All Images")
    print("="*60)
    
    # Get all image paths
    image_dict = get_all_image_paths(dataset_root)
    
    print(f"Found {len(image_dict)} folders")
    total_images = sum(len(paths) for paths in image_dict.values())
    print(f"Total images found: {total_images}")
    
    # Count spectral band images (excluding dark images)
    spectral_images = sum(1 for paths in image_dict.values() 
                         for p in paths if not is_dark_image(p))
    dark_images = total_images - spectral_images
    print(f"  - Spectral band images: {spectral_images}")
    print(f"  - Dark images (000nm): {dark_images}")
    
    print("\n" + "="*60)
    print("STEP 3: Select Crop Region")
    print("="*60)
    
    # Select a sample image for cropping (skip dark image - 000nm)
    sample_folder = next(iter(image_dict))
    sample_paths = [p for p in image_dict[sample_folder] if not is_dark_image(p)]
    
    if not sample_paths:
        print("Error: No spectral band images found for crop selection")
        return
    
    # Use the first spectral band image (should be after 000nm)
    sample_image_path = sample_paths[0]
    print(f"Using sample image: {Path(sample_image_path).name}")
    print(f"Full path: {sample_image_path}")
    
    # Show image and get crop point
    crop_center = show_sample_image_and_get_crop_point(sample_image_path, dark_image)
    
    if crop_center:
        print(f"\nCrop center selected at: {crop_center}")
        print(f"Crop size: 100x100 pixels")
        
        confirm = input("\nProceed with processing all images? (y/n): ")
        if confirm.lower() != 'y':
            print("Processing cancelled.")
            return
        
        print("\n" + "="*60)
        print("STEP 4: Processing All Images")
        print("="*60)
        
        # Process all images: dark current reduction + cropping
        crop_and_save_all_images(image_dict, crop_center, save_root, 
                                 dark_image=dark_image, original_root=dataset_root)
        
        print(f"\nAll preprocessed images saved to:\n{save_root}")
        
        # Summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Input folder: {dataset_root}")
        print(f"Output folder: {save_root}")
        print(f"Dark current reduction: {'Applied' if dark_image is not None else 'Not applied'}")
        print(f"Crop size: 100x100 pixels")
        print(f"Crop center: {crop_center}")
        print("="*60)
    else:
        print("No crop point selected. Exiting.")

if __name__ == "__main__":
    main()