import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ArtisticTransformer:
    def __init__(self):
        """Initialize the transformer with default parameters"""
        self.image = None
        self.height = None
        self.width = None
    
    def load_image(self, image_path):
        """Load and prepare the image for processing"""
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        return self
    
    def impressionist_style(self, brush_size=15):
        """Transform image to impressionist style
        
        Args:
            brush_size (int): Size of the brush effect (Gaussian blur kernel)
        """
        # Apply Gaussian blur to simulate brush strokes
        blurred = cv2.GaussianBlur(self.image, (brush_size, brush_size), 0)
        
        # Enhance edges to create painting-like effect
        edges = cv2.Canny(cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY), 50, 150)
        edges = cv2.dilate(edges, None)
        
        # Combine effects
        mask = edges > 0
        result = np.copy(self.image)
        result[mask] = blurred[mask]
        
        # Enhance color saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # Increase saturation
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def cubist_style(self, num_segments=100):
        """Transform image to cubist style
        
        Args:
            num_segments (int): Number of geometric segments
        """
        # Convert to LAB color space for better segmentation
        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        
        # Apply segmentation
        segments = cv2.ximgproc.createSuperpixelSLIC(lab, algorithm=cv2.ximgproc.SLIC, 
                                                    region_size=50)
        segments.iterate(10)
        labels = segments.getLabels()
        
        # Create geometric effect
        result = np.zeros_like(self.image)
        for label in range(num_segments):
            mask = labels == label
            if mask.any():
                # Calculate average color for segment
                color = self.image[mask].mean(axis=0)
                result[mask] = color
                
                # Add geometric edges
                contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 0, 0), 1)
        
        return result
    
    def pointillist_style(self, dot_size=5, spacing=7):
        """Transform image to pointillist style
        
        Args:
            dot_size (int): Size of dots
            spacing (int): Space between dots
        """
        # Create blank canvas
        result = np.zeros_like(self.image)
        
        # Create dots
        for y in range(0, self.height, spacing):
            for x in range(0, self.width, spacing):
                # Get color from original image
                color = self.image[y:y+spacing, x:x+spacing].mean(axis=(0,1))
                
                # Draw colored dot
                cv2.circle(result, 
                          center=(x + spacing//2, y + spacing//2),
                          radius=dot_size,
                          color=tuple(map(int, color)),
                          thickness=-1)
        
        return result
    
    def display_results(self, transformed_image, title="Transformed Image"):
        """Display original and transformed images side by side"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        plt.imshow(self.image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(transformed_image)
        plt.title(title)
        plt.axis('off')
        
        plt.show()