from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import os
from langchain.output_parsers.json import parse_json_markdown
from flask import jsonify

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

STEP_TYPE_MAP = {
    "convert_to_bw": ("convert_to_bw", "BW"),
    "detect_objects_tool": ("object_detection", "detection"),
    "general_qa_tool": ("general_Q&A", "QA"),
    "sam_segment_tool": ("segmentation", "segmentation"),
    "caption_image_tool": ("image_captioning", "captioning"),
    "ocr_image": ("OCR", "OCR"),
    "diffusion_inpainting_tool": ("image_inpainting", "inpainting"),
}

def get_info_from_step(step_name, step_response, prompt_id, suffix):
    url = ''
    image_format = ''
    filename = ''
    step_contains_image = False
    if 'inpainted_image_url' in step_response or 'annotated_image_url' in step_response or 'mask_url' in step_response or 'bw_image_url' in step_response or 'merged_mask_url' in step_response: 
        step_contains_image = True


    if  step_contains_image:
        url = None
        if 'annotated_image_url' in step_response:
            url = step_response.get('annotated_image_url')
        elif 'merged_mask_url' in step_response:
            url = step_response.get('merged_mask_url')
        elif 'bw_image_url' in step_response:
            url = step_response.get('bw_image_url')
        elif 'mask_url' in step_response:
            url = step_response.get('mask_url')
        elif 'inpainted_image_url' in step_response:
            url = step_response.get('inpainted_image_url')

        image_format = url.split('.')[-1]
        filename =  f"{prompt_id}_{suffix}.{image_format}"
        resource_type = "image"
    else:
        resource_type = "text"

    return url, image_format, filename, resource_type

def build_sets_list(step_name, step_response, prompt_id, suffix, user_id, steps):
    url, image_format, filename, resource_type = get_info_from_step(step_name, step_response, prompt_id, suffix)
        
    steps.append({
        "step_type": step_name,
        "public_id": f"{user_id}/steps/{prompt_id}_{suffix}",
        "filename":  filename,
        "url": url,
        "resource_type": resource_type,
        "format": image_format
    })
    return steps

def format_endpoint_response(response, user_id, prompt_id):
    """
    Format the response from the endpoint to be sent to the client.
    """

    steps = []
    for i, step in enumerate(response['intermediate_steps']):
        if(step[0].tool == 'None'):
            continue
        step_type = STEP_TYPE_MAP.get(step[0].tool)
        if(step_type is None):
            continue
        step_response = json_parser(step[1])
        suffix = step_type[1]
        step_name = step_type[0]

        # Only set to "output" if it's the last step
        if i == len(response['intermediate_steps']) - 1:
            step_name = "output"
            suffix = "output"

        steps = build_sets_list(step_name, step_response, prompt_id, suffix, user_id, steps)




    final_response = json_parser(response['output'])
    

    result_data = {
            "final_response": final_response,
            "steps": steps
        }
    return result_data