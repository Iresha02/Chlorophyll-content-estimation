import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

class SuperpixelDataMatrixBuilder:
    """
    Constructs data matrix using superpixel approach for multispectral imaging.
    Based on the document's methodology (Section III-D).
    """
    
    def __init__(self, preprocessed_folder, block_size=10):
        """
        Initialize the data matrix builder.
        
        Args:
            preprocessed_folder: Path to folder containing preprocessed images
            block_size: Size of superpixel blocks (default: 10x10)
        """
        self.preprocessed_folder = Path(preprocessed_folder)
        self.block_size = block_size
        
        # Define spectral bands in order (excluding 000nm dark image)
        self.spectral_bands = [
            '365nm', '405nm', '473nm', '530nm', '575nm',
            '621nm', '660nm', '735nm', '770nm', '830nm', 
            '850nm', '890nm', '940nm'
        ]
        
        self.data_matrix = None
        self.labels = []
        self.sample_info = []
        
    def get_sample_folders(self):
        """Get all sample folders from preprocessed directory."""
        sample_folders = []
        for item in self.preprocessed_folder.iterdir():
            if item.is_dir():
                sample_folders.append(item)
        
        # Sort folders for consistent ordering
        sample_folders = sorted(sample_folders, key=lambda x: x.name)
        
        print(f"Found {len(sample_folders)} sample folders")
        return sample_folders
    
    def load_spectral_images(self, sample_folder):
        """
        Load all spectral band images for a single sample.
        
        Args:
            sample_folder: Path to folder containing spectral band images
            
        Returns:
            List of 13 images (one per spectral band), or None if incomplete
        """
        images = []
        missing_bands = []
        
        for band in self.spectral_bands:
            # Search for image file with this wavelength in name
            image_path = None
            for file in sample_folder.iterdir():
                if band.lower() in file.name.lower() and file.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                    image_path = file
                    break
            
            if image_path and image_path.exists():
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                else:
                    missing_bands.append(band)
                    print(f"  Warning: Could not read {image_path}")
            else:
                missing_bands.append(band)
                print(f"  Warning: Missing {band} in {sample_folder.name}")
        
        if missing_bands:
            print(f"  Missing bands for {sample_folder.name}: {missing_bands}")
            return None
        
        # Verify all images have same dimensions
        shapes = [img.shape for img in images]
        if len(set(shapes)) > 1:
            print(f"  Warning: Inconsistent image sizes in {sample_folder.name}: {shapes}")
            return None
        
        return images
    
    def create_superpixels(self, images):
        """
        Create superpixels from spectral band images.
        
        Args:
            images: List of 13 spectral band images
            
        Returns:
            Numpy array of shape (num_superpixels, 13)
        """
        if not images:
            return None
        
        h, w = images[0].shape
        num_blocks_h = h // self.block_size
        num_blocks_w = w // self.block_size
        
        superpixels = []
        
        # Iterate over each block position
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # Calculate block boundaries
                y_start = i * self.block_size
                y_end = y_start + self.block_size
                x_start = j * self.block_size
                x_end = x_start + self.block_size
                
                # Extract average intensity for each spectral band
                superpixel_row = []
                for band_img in images:
                    block = band_img[y_start:y_end, x_start:x_end]
                    avg_intensity = np.mean(block)
                    superpixel_row.append(avg_intensity)
                
                superpixels.append(superpixel_row)
        
        return np.array(superpixels)
    
    def build_data_matrix(self, extract_label_from_folder_name=True):
        """
        Build complete data matrix from all samples.
        
        Args:
            extract_label_from_folder_name: If True, tries to extract percentage from folder name
            
        Returns:
            Numpy array of shape (total_superpixels, 13)
        """
        print("="*60)
        print("BUILDING DATA MATRIX USING SUPERPIXELS")
        print("="*60)
        
        sample_folders = self.get_sample_folders()
        
        if not sample_folders:
            print("Error: No sample folders found!")
            return None
        
        all_superpixels = []
        
        print(f"\nProcessing {len(sample_folders)} samples...")
        print(f"Block size: {self.block_size}×{self.block_size}")
        print(f"Spectral bands: {len(self.spectral_bands)}")
        print("-"*60)
        
        for idx, sample_folder in enumerate(sample_folders, 1):
            print(f"\n[{idx}/{len(sample_folders)}] Processing: {sample_folder.name}")
            
            # Load all spectral band images
            images = self.load_spectral_images(sample_folder)
            
            if images is None:
                print(f"  ⚠ Skipping {sample_folder.name} (incomplete data)")
                continue
            
            # Create superpixels
            superpixels = self.create_superpixels(images)
            
            if superpixels is not None:
                num_superpixels = superpixels.shape[0]
                print(f"  ✓ Created {num_superpixels} superpixels")
                
                # Add to master list
                all_superpixels.append(superpixels)
                
                # Store sample information
                self.sample_info.append({
                    'folder_name': sample_folder.name,
                    'num_superpixels': num_superpixels,
                    'index_start': len(self.labels),
                    'index_end': len(self.labels) + num_superpixels
                })
                
                # Extract and store labels (repeat for each superpixel)
                if extract_label_from_folder_name:
                    label = self.extract_label(sample_folder.name)
                else:
                    label = sample_folder.name
                
                self.labels.extend([label] * num_superpixels)
        
        if not all_superpixels:
            print("\nError: No valid samples processed!")
            return None
        
        # Stack all superpixels vertically
        self.data_matrix = np.vstack(all_superpixels)
        
        print("\n" + "="*60)
        print("DATA MATRIX CONSTRUCTION COMPLETE")
        print("="*60)
        print(f"Data matrix shape: {self.data_matrix.shape}")
        print(f"  Rows (total superpixels): {self.data_matrix.shape[0]}")
        print(f"  Columns (spectral bands): {self.data_matrix.shape[1]}")
        print(f"Total samples processed: {len(self.sample_info)}")
        print(f"Unique labels: {len(set(self.labels))}")
        print("="*60)
        
        return self.data_matrix
    
    def extract_label(self, folder_name):
        """
        Extract label (e.g., adulteration percentage) from folder name.
        Tries to find numbers in the folder name.
        
        Args:
            folder_name: Name of the sample folder
            
        Returns:
            Extracted label or original folder name
        """
        import re
        
        # Try to find percentage pattern (e.g., "10%", "10percent", "10_percent")
        percentage_match = re.search(r'(\d+)\s*%|(\d+)\s*percent', folder_name.lower())
        if percentage_match:
            return int(percentage_match.group(1) or percentage_match.group(2))
        
        # Try to find any number
        number_match = re.search(r'(\d+)', folder_name)
        if number_match:
            return int(number_match.group(1))
        
        # Return folder name as-is if no number found
        return folder_name
    
    def save_data_matrix(self, output_folder=None):
        """
        Save data matrix and labels to files.
        
        Args:
            output_folder: Path to save files (default: same as preprocessed folder)
        """
        if self.data_matrix is None:
            print("Error: Data matrix not built yet. Run build_data_matrix() first.")
            return
        
        if output_folder is None:
            output_folder = self.preprocessed_folder.parent / "DataMatrix"
        else:
            output_folder = Path(output_folder)
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save data matrix as numpy array
        np.save(output_folder / "data_matrix.npy", self.data_matrix)
        print(f"\n✓ Data matrix saved to: {output_folder / 'data_matrix.npy'}")
        
        # Save labels
        np.save(output_folder / "labels.npy", np.array(self.labels))
        print(f"✓ Labels saved to: {output_folder / 'labels.npy'}")
        
        # Save as CSV for easy inspection
        df = pd.DataFrame(self.data_matrix, columns=self.spectral_bands)
        df['label'] = self.labels
        df.to_csv(output_folder / "data_matrix.csv", index=False)
        print(f"✓ Data matrix (CSV) saved to: {output_folder / 'data_matrix.csv'}")
        
        # Save sample information
        sample_info_df = pd.DataFrame(self.sample_info)
        sample_info_df.to_csv(output_folder / "sample_info.csv", index=False)
        print(f"✓ Sample info saved to: {output_folder / 'sample_info.csv'}")
        
        print(f"\nAll files saved to: {output_folder}")
    
    def visualize_spectral_signatures(self, num_samples=5):
        """
        Visualize spectral signatures for a few samples.
        
        Args:
            num_samples: Number of samples to visualize
        """
        if self.data_matrix is None:
            print("Error: Data matrix not built yet.")
            return
        
        # Get unique labels
        unique_labels = sorted(set(self.labels))
        
        # Select samples from different classes
        samples_to_plot = []
        for label in unique_labels[:num_samples]:
            # Find first superpixel of this label
            idx = self.labels.index(label)
            samples_to_plot.append((label, self.data_matrix[idx]))
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        wavelengths = [int(band.replace('nm', '')) for band in self.spectral_bands]
        
        for label, signature in samples_to_plot:
            plt.plot(wavelengths, signature, marker='o', label=f'Label: {label}')
        
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Average Intensity', fontsize=12)
        plt.title('Spectral Signatures of Different Samples', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """Print statistics about the data matrix."""
        if self.data_matrix is None:
            print("Error: Data matrix not built yet.")
            return
        
        print("\n" + "="*60)
        print("DATA MATRIX STATISTICS")
        print("="*60)
        
        print(f"\nShape: {self.data_matrix.shape}")
        print(f"\nIntensity Statistics:")
        print(f"  Min: {self.data_matrix.min():.2f}")
        print(f"  Max: {self.data_matrix.max():.2f}")
        print(f"  Mean: {self.data_matrix.mean():.2f}")
        print(f"  Std: {self.data_matrix.std():.2f}")
        
        print(f"\nLabel Distribution:")
        from collections import Counter
        label_counts = Counter(self.labels)
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} superpixels")
        
        print(f"\nSpectral Band Statistics:")
        for i, band in enumerate(self.spectral_bands):
            band_data = self.data_matrix[:, i]
            print(f"  {band}: mean={band_data.mean():.2f}, std={band_data.std():.2f}")
        
        print("="*60)


def main():
    """Main execution function."""
    
    # Configuration
    preprocessed_folder = r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\Filtered"
    block_size = 10  # 10x10 superpixels as per document
    
    print("="*60)
    print("SUPERPIXEL DATA MATRIX CONSTRUCTION")
    print("Based on Document Section III-D")
    print("="*60)
    print(f"\nPreprocessed folder: {preprocessed_folder}")
    print(f"Block size: {block_size}×{block_size}")
    print(f"Expected superpixels per 100×100 image: {(100//block_size)**2}")
    
    # Create builder
    builder = SuperpixelDataMatrixBuilder(
        preprocessed_folder=preprocessed_folder,
        block_size=block_size
    )
    
    # Build data matrix
    data_matrix = builder.build_data_matrix(extract_label_from_folder_name=True)
    
    if data_matrix is not None:
        # Show statistics
        builder.get_statistics()
        
        # Save data matrix
        builder.save_data_matrix()
        
        # Visualize spectral signatures
        print("\nGenerating spectral signature visualization...")
        builder.visualize_spectral_signatures(num_samples=5)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("\nNext steps:")
        print("1. Load data_matrix.npy and labels.npy")
        print("2. Split into train/test sets (75%/25%)")
        print("3. Apply PCA or LDA for feature extraction")
        print("4. Train classification models")
        print("="*60)
    else:
        print("\nError: Failed to build data matrix.")


if __name__ == "__main__":
    main()