from langchain_core.tools import tool  
from utils.utils import json_parser
from PIL import Image
from urllib.request import urlopen
import io
import requests
import json

@tool
def segment_image_tool(input: str) -> str:
    """
    Segment objects in an image based on a text prompt by calling the /segment API endpoint.
    
    The input should be a JSON-formatted string with:
      - "image_url": str, the URL of the image to segment.
      - "prompt": str, a text description of the object to segment, it must be very specific (e.g., "black dog").
      
    Returns:
      str: A JSON-formatted string with segmentation results containing:
          - "success": bool,
          - "prompt": str,
          - "original_image_url": str,
          - "mask_url": str.
    """
    # Parse the input JSON and extract the required fields
    try:
        data = json_parser(input)
        image_url = data.get("image_url")
        prompt = data.get("prompt")
        if not image_url or not prompt:
            raise ValueError("Missing 'image_url' or 'prompt' in input.")
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with keys 'image_url' and 'prompt'. Error: {e}")
    
    # Download the image from the provided URL
    try:
        image_data = urlopen(image_url).read()
    except Exception as e:
        raise ValueError(f"Could not download image from {image_url}: {e}")
    
    # Open the image using Pillow to determine its format (e.g., PNG, JPEG)
    try:
        image = Image.open(io.BytesIO(image_data))
        image_format = image.format if image.format else "PNG"
    except Exception as e:
        raise ValueError(f"Could not open image data: {e}")
    
    # Build a filename and set content type based on the image format
    ext = image_format.lower()
    filename = f"temp_image.{ext}"
    
    # Define the API endpoint URL (adjust if your endpoint is hosted elsewhere)
    api_url = "http://localhost:8000/segment"
    
    # Prepare files and form data for the POST request
    files = {"file": (filename, image_data, f"image/{ext}")}
    form_data = {"prompt": prompt}
    
    try:
        response = requests.post(api_url, files=files, data=form_data)
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to call segmentation API: {e}")
    
    # Parse and return the JSON response as a string
    try:
        result = response.json()
    except Exception as e:
        raise ValueError(f"API did not return valid JSON: {e}")
    
    return json.dumps(result)


@tool
def detect_objects_tool(input: str) -> str:
    """
    Detect objects in an image based on a text prompt by calling the /detect API endpoint.
    
    The input should be a JSON-formatted string with:
      - "image_url": str, the URL of the image to process.
      - "prompt": str, a text description of the objects to detect (e.g., "people").
      
    Returns:
      str: A JSON-formatted string with detection results containing:
          - "success": bool,
          - "prompt": str,
          - "original_image_url": str,
          - "bounding_boxes": list, list of bounding boxes [x1, y1, x2, y2],
          - "centroids": list, list of centroids [cx, cy] for each box,
          - "labels": list, list of labels for each detected object.
    """
    # Parse the input JSON and extract the required fields
    try:
        data = json_parser(input)
        image_url = data.get("image_url")
        prompt = data.get("prompt")
        if not image_url or not prompt:
            raise ValueError("Missing 'image_url' or 'prompt' in input.")
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with keys 'image_url' and 'prompt'. Error: {e}")
    
    # Download the image from the provided URL
    try:
        image_data = urlopen(image_url).read()
    except Exception as e:
        raise ValueError(f"Could not download image from {image_url}: {e}")
    
    # Open the image using Pillow to determine its format (e.g., PNG, JPEG)
    try:
        image = Image.open(io.BytesIO(image_data))
        image_format = image.format if image.format else "PNG"
    except Exception as e:
        raise ValueError(f"Could not open image data: {e}")
    
    # Build a filename and set content type based on the image format
    ext = image_format.lower()
    filename = f"temp_image.{ext}"
    
    # Define the API endpoint URL (adjust if your endpoint is hosted elsewhere)
    api_url = "http://localhost:8000/detect"
    
    # Prepare files and form data for the POST request
    files = {"file": (filename, image_data, f"image/{ext}")}
    form_data = {"prompt": prompt}
    
    # Call the detection API endpoint
    try:
        response = requests.post(api_url, files=files, data=form_data)
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to call detection API: {e}")
    
    # Parse and return the JSON response as a string
    try:
        result = response.json()
    except Exception as e:
        raise ValueError(f"API did not return valid JSON: {e}")
    
    return json.dumps(result)