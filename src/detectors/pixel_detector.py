"""
Pixel Detector Module

Performs Error Level Analysis (ELA) and noise variance detection.
"""

from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
from io import BytesIO


class PixelDetector:
    """Detects fraud indicators through pixel-level analysis."""
    
    def analyze_ela(self, image):
        """
        Perform Error Level Analysis (ELA) on an image.
        
        ELA works by re-saving the image at a lower quality and comparing
        it to the original. Edited areas show more difference than unedited areas.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image: The ELA heatmap showing differences
        """
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save original image to memory at 90% quality
        original_buffer = BytesIO()
        image.save(original_buffer, format='JPEG', quality=90)
        original_buffer.seek(0)
        
        # Load the re-saved image
        resaved_image = Image.open(original_buffer)
        resaved_image = resaved_image.convert('RGB')
        
        # Calculate absolute difference
        diff = ImageChops.difference(image, resaved_image)
        
        # Enhance brightness by factor of 20 to make differences more visible
        enhancer = ImageEnhance.Brightness(diff)
        heatmap = enhancer.enhance(20.0)
        
        return heatmap
    
    def analyze_noise(self, image):
        """
        Analyze noise variance in an image to detect blur/smoothing.
        
        Edited areas often have lower noise variance due to smoothing/blurring.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: {
                'variance': float,
                'flags': list of str,
                'findings': str
            }
        """
        findings = []
        flags = []
        
        # Convert PIL Image to numpy array
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        img_array = np.array(gray_image)
        
        # Calculate Laplacian variance using OpenCV
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        variance = laplacian.var()
        
        # Check for unusually low variance (indicates blur/smoothing)
        if variance < 100:
            flags.append('Potential Image Smoothing Detected')
            findings.append(
                f'Low noise variance ({variance:.2f}) detected. '
                'This may indicate edited or smoothed areas consistent with digital manipulation.'
            )
        else:
            findings.append(
                f'Normal noise variance ({variance:.2f}) detected. '
                'No significant blur or smoothing indicators present.'
            )
        
        return {
            'variance': float(variance),
            'flags': flags,
            'findings': ' '.join(findings)
        }
