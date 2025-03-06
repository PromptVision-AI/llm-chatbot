from langchain_core.tools import tool  
from utils.utils import json_parser
from PIL import Image
from utils.utils import upload_image_to_cloudinary
import os
import io
import tempfile
from urllib.request import urlopen


@tool
def convert_to_bw(input: str) -> str:
    """
    Convert a remote image (specified by a URL) to black and white (grayscale) 
    and upload it to Cloudinary.

    The input should be a JSON-formatted string with:
      - "image_url": str, the URL of the image to convert.

    Returns:
      str: The URL of the modified image stored in Cloudinary.
    """
    try:
        data = json_parser(input)
        image_url = data.get("image_url")
        if not image_url:
            raise ValueError("Missing 'image_url' in input.")
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with key 'image_url'. Error: {e}")

    # 1. Fetch the image data from the remote URL
    try:
        image_data = urlopen(image_url).read()
    except Exception as e:
        raise ValueError(f"Could not download image from {image_url}: {e}")

    # 2. Open the image in memory using Pillow
    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Could not open image data: {e}")

    # 3. Convert the image to grayscale (black and white)
    bw_image = image.convert("L")

    # 4. Save the modified image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        bw_image.save(temp_path, format="PNG")

    # 5. Upload the image to Cloudinary
    try:
        cloud_url = upload_image_to_cloudinary(temp_path)
    except Exception as e:
        # Clean up the temp file if upload fails
        os.remove(temp_path)
        raise ValueError(f"Failed to upload image to Cloudinary: {e}")

    # 6. Remove the local temporary file
    os.remove(temp_path)

    # 7. Return the Cloudinary URL
    return cloud_url