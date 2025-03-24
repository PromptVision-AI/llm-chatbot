from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import os
from langchain.output_parsers.json import parse_json_markdown


# Load the .env file
load_dotenv()

# Retrieve the variables
CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")




# Configure Cloudinary with your credentials
cloudinary.config( 
  cloud_name = CLOUDINARY_NAME,        # Replace with your Cloudinary cloud name
  api_key = CLOUDINARY_API_KEY,              # Replace with your Cloudinary API key
  api_secret = CLOUDINARY_API_SECRET         # Replace with your Cloudinary API secret
)

def json_parser(text):
    # Remove any extraneous wrapping quotes
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    try:
        obj = parse_json_markdown(text)
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        return None
    return obj




def configure_cloudinary():
    """
    Configure Cloudinary using environment variables.
    Make sure you have CLOUDINARY_NAME, CLOUDINARY_API_KEY, 
    and CLOUDINARY_API_SECRET set in your .env file.
    """
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )

def upload_image_to_cloudinary(image_file):
    """
    Uploads an image file (from Flask's request.files) to Cloudinary 
    and returns the secure URL.

    Args:
        image_file (FileStorage): The file object from request.files

    Returns:
        str: The secure URL of the uploaded image
    """
    result = cloudinary.uploader.upload(image_file)
    secure_url = result.get("secure_url")
    return secure_url