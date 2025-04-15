from langchain_core.tools import tool  
from utils.utils import json_parser, upload_image_to_cloudinary
from PIL import Image
from urllib.request import urlopen
import io
import json
from ultralytics import SAM
import tempfile
import os
import numpy as np
import cv2

def apply_morphological_operations(mask, kernel_size=8):
    """
    Apply morphological operations to improve mask quality.
    
    Args:
        mask (numpy.ndarray): Binary mask
        kernel_size (int): Size of the kernel for morphological operations
        
    Returns:
        numpy.ndarray: Processed binary mask
    """
    if mask is None:
        return None
    
    
    # Create kernels of different sizes for different operations
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    main_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Convert to uint8 and scale to 0-255 range for better precision
    mask = mask.astype(np.uint8).squeeze() * 255
    
    # 1. Opening (erosion then dilation) - removes small noise/isolated pixels
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=3)
    
    # 2. Closing (dilation then erosion) - fills small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, main_kernel, iterations=3)
    
    # 3. Remove small isolated regions
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Remove small components (adjust 50 threshold as needed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 20:  # Minimum area threshold
            mask[labels == i] = 0
    
    # 4. Final dilation to expand the mask slightly
    mask = cv2.dilate(mask, main_kernel, iterations=3)
    
    # 5. Optional: Edge smoothing with Gaussian blur followed by threshold
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask > 0  # Convert back to boolean mask

def merge_masks(masks):
    """
    Merge multiple binary masks into a single mask using logical OR operation.
    
    Args:
        masks (list): List of binary masks as numpy arrays
        
    Returns:
        numpy.ndarray: Merged binary mask
    """
    if not masks:
        return None
    
    # Convert masks to numpy arrays if they aren't already
    mask_arrays = [np.array(mask) for mask in masks]
    
    # Merge masks using logical OR
    merged_mask = np.zeros_like(mask_arrays[0])
    for mask in mask_arrays:
        merged_mask = np.logical_or(merged_mask, mask)
    
    return merged_mask

@tool
def sam_segment_tool(input: str) -> str:
    """
    Segment objects in an image using SAM (Segment Anything Model) based on bounding boxes. A requirement for this tool is that the bounding boxes must be provided in the input.
    
    The input should be a JSON-formatted string with:
      - "image_url": str, the Cloudinary URL of the image to segment.
      - "bounding_boxes": list, list of bounding boxes in format [x1, y1, x2, y2].
      
    Returns:
      str: A JSON-formatted string with segmentation results containing:
          - "success": bool,
          - "original_image_url": str,
          - "merged_mask_url": str, URL of the merged mask image in Cloudinary.
    """
    # Parse the input JSON and extract the required fields
    try:
        data = json_parser(input)
        image_url = data.get("image_url")
        bounding_boxes = data.get("bounding_boxes")
        if not image_url or not bounding_boxes:
            raise ValueError("Missing 'image_url' or 'bounding_boxes' in input.")
        if not isinstance(bounding_boxes, list):
            raise ValueError("'bounding_boxes' must be a list.")
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with keys 'image_url' and 'bounding_boxes'. Error: {e}")
    
    # Download the image from the provided Cloudinary URL
    try:
        image_data = urlopen(image_url).read()
    except Exception as e:
        raise ValueError(f"Could not download image from {image_url}: {e}")
    
    # Create a temporary file to store the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name
    
    try:
        # Initialize SAM model
        model = SAM("tools/sam/sam2.1_l.pt")
        # Run SAM inference with bounding boxes
        results = model(temp_path, bboxes=bounding_boxes)
        
        # Extract masks from results
        masks = []
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                # Convert masks to binary format
                for mask in result.masks:
                    masks.append(mask.data.cpu().numpy().tolist())
        
        # Merge all masks into one
        merged_mask = merge_masks(masks)

        if merged_mask is not None:
            merged_mask = apply_morphological_operations(merged_mask, kernel_size=7)

        
        # Convert merged mask to PIL Image and save to temporary file
        if merged_mask is not None:
            mask_image = Image.fromarray((np.squeeze(merged_mask) * 255).astype(np.uint8))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as mask_file:
                mask_path = mask_file.name
                mask_image.save(mask_path, format='PNG')
                
                # Upload merged mask to Cloudinary
                try:
                    merged_mask_url = upload_image_to_cloudinary(mask_path)
                except Exception as e:
                    raise ValueError(f"Failed to upload merged mask to Cloudinary: {e}")
                finally:
                    # Clean up temporary mask file
                    try:
                        os.unlink(mask_path)
                    except:
                        pass
        else:
            merged_mask_url = None
        
        # Prepare response
        response = {
            "success": True,
            "original_image_url": image_url,
            "merged_mask_url": merged_mask_url
        }
        
    except Exception as e:
        raise ValueError(f"Failed to run SAM segmentation: {e}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    return json.dumps(response) 