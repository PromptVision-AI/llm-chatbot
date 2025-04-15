"""
Optical Character Recognition (OCR) tool for extracting text from images.
"""
import json
from langchain_core.tools import tool
from utils.utils import json_parser
from PIL import Image
import requests
from io import BytesIO
import pytesseract


@tool
def ocr_image(input: str) -> str:
    """
    Extract text from an image using Optical Character Recognition (OCR).
    
    The input should be a JSON-formatted string with:
      - "image_url": str, the URL of the image to extract text from.
    
    Returns:
      str: A JSON-formatted string with the extracted text:
          - "extracted_text": str, the extracted text from the image.
    """
    try:
        data = json_parser(input)
        image_url = data.get("image_url")
        if not image_url:
            raise ValueError("Missing 'image_url' in input.")
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with key 'image_url'. Error: {e}")
    
    # 1. Download the image from the URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        raise ValueError(f"Could not download or open image from {image_url}: {e}")
    
    # 2. Use pytesseract to extract text
    try:
        
        extracted_text = pytesseract.image_to_string(image)
        text = extracted_text.strip()
        
        if not text:
            return "No text could be extracted from the image."
        
        return json.dumps({"extracted_text": text})
    except Exception as e:
        
        raise ValueError(f"OCR processing failed: {e}")
