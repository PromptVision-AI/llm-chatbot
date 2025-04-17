import os
import requests
from io import BytesIO
import json
import torch
import numpy as np
from PIL import Image
# import matplotlib # No longer needed unless for saving plots, which we aren't doing
# matplotlib.use('Agg') # Use non-interactive backend for servers
# import matplotlib.pyplot as plt # No longer needed
import tempfile
import shutil # For potential cleanup on error

from diffusers import (
    AutoPipelineForImage2Image, # As requested
    ControlNetModel,
    # StableDiffusionControlNetInpaintPipeline # Not used in this specific structure
)
from langchain_core.tools import tool

# Assuming utility functions are available from your project structure
# Make sure these imports work in your actual project setup
from utils.utils import json_parser, upload_image_to_cloudinary

# --- Configuration Constants (Consider making these configurable if needed) ---
# This directory should contain the VAE, UNet, text_encoder, etc. for the base model.
BASE_MODEL_SAVE_DIRECTORY = "tools/diffusion/saved_pipeline" # Or "path/to/your/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_inpaint"
# Default generation parameters (used by the img2img pipeline)
DEFAULT_STRENGTH = 0.6  # Controls how much the init_image is changed
DEFAULT_CFG_SCALE = 7.5 # Controls prompt adherence
DEFAULT_NUM_STEPS = 50  # Number of diffusion steps


# --- Helper Functions ---
def download_image(url):
    """Downloads an image from a URL and returns a PIL Image."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise HTTP errors
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Downloaded image from {url}")
        return image
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download image from {url}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to process image data from {url}: {e}")

def make_inpaint_condition(init_image, mask_image):
    """Prepares the 'control image' format required by some ControlNet models."""
    # Note: This control image might not be used by the AutoPipelineForImage2Image
    try:
        image_np = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        if image_np.shape[:2] != mask_np.shape[:2]:
            raise ValueError(f"Image ({image_np.shape[:2]}) and mask ({mask_np.shape[:2]}) must have the same height and width.")

        # Standard ControlNet inpainting condition: mask pixels are set to -1
        image_np[mask_np > 0.5] = -1.0
        # Add batch dimension and channels-first format (B, C, H, W)
        control_image_np = np.expand_dims(image_np, 0).transpose(0, 3, 1, 2)
        control_image = torch.from_numpy(control_image_np)
        print("Prepared ControlNet condition image format.")
        return control_image
    except Exception as e:
        raise ValueError(f"Failed to create inpaint condition: {e}")

def save_and_upload_image(image: Image.Image, step_name="image"):
    """Saves a PIL image temporarily, uploads it via helper, returns URL, cleans up."""
    temp_path = None
    try:
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, format="PNG")
            print(f"Saved intermediate image for upload: {temp_path}")

        # Upload the saved image file
        print(f"Attempting to upload {step_name} from {temp_path}")
        # This assumes upload_image_to_cloudinary takes a file path
        image_url = upload_image_to_cloudinary(temp_path)
        print(f"Upload successful for {step_name}: {image_url}")
        return image_url
    except Exception as e:
        print(f"Error during save/upload for {step_name}: {e}")
        # Re-raise to allow the main error handler to catch it
        raise ValueError(f"Failed to save or upload {step_name}: {e}")
    finally:
        # Ensure cleanup of the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_e:
                # Log warning if cleanup fails but don't crash the tool
                print(f"Warning: Failed to delete temporary file {temp_path}: {cleanup_e}")

# --- Tool Definition ---
@tool
def diffusion_inpainting_tool(input: str) -> str:
    """
    Modifies an input image based on a text prompt using a standard image-to-image pipeline.
    Accepts an image, mask, and prompt, but primarily uses the image and prompt for generation. 
    The final image is uploaded to Cloudinary.

    The input should be a JSON-formatted string with:
      - "image_url": str, URL of the initial image.
      - "mask_url": str, URL of the mask image, this must be obtained using segment_image_tool 
      - "prompt": str, text description to guide image modification.

    Returns:
      str: A JSON-formatted string containing:
        - "success": bool, True if completed successfully, False otherwise.
        - "final_image_url": str, Cloudinary URL of the final generated image (if successful).
        - "error": str, (Optional) Error message if success is False.
    """
    # --- Verify Configuration ---
    if not os.path.exists(BASE_MODEL_SAVE_DIRECTORY):
        error_msg = f"Configuration Error: Base model directory not found at '{BASE_MODEL_SAVE_DIRECTORY}'. Please configure the path."
        print(f"Error: {error_msg}")
        return json.dumps({"success": False, "error": error_msg})

    controlnet = None # Initialize for potential cleanup in finally block
    img2img_pipeline = None
    try:
        # --- 1. Parse Input ---
        print("Parsing input JSON...")
        data = json_parser(input) # Assumes this helper exists
        image_url = data.get("image_url")
        mask_url = data.get("mask_url") # Parsed but likely unused by pipeline
        prompt = data.get("prompt")

        # Validate required inputs
        if not all([image_url, mask_url, prompt]):
            raise ValueError("Missing required input: 'image_url', 'mask_url', or 'prompt'.")
        print("Input parsed successfully.")

        # Determine compute device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # --- 2. Load Models (as per user's specified structure) ---
        print(f"Loading ControlNet model: {CONTROLNET_MODEL_ID} (Note: May not be used by pipeline)...")
        # Load ControlNet model (even if potentially unused by the chosen pipeline)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(device)

        print(f"Loading Image2Image pipeline from: {BASE_MODEL_SAVE_DIRECTORY}")
        # Load the standard Image-to-Image pipeline
        img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
            BASE_MODEL_SAVE_DIRECTORY,
            torch_dtype=torch.float16
            ).to(device)
        print("Models loaded successfully.")

        # --- 3. Prepare Images ---
        print("Downloading initial image and mask...")
        init_image = download_image(image_url)
        mask_image = download_image(mask_url) # Downloaded but likely unused

        # Ensure mask is binary (though likely unused)
        mask_image = mask_image.convert("L").point(lambda p: 255 if p > 127 else 0, mode='L')

        # Prepare ControlNet condition image format (though likely unused by this pipeline)
        print("Preparing ControlNet condition image format (Note: May not be used by pipeline)...")
        control_image = make_inpaint_condition(init_image, mask_image)

        # Resize init_image if dimensions aren't suitable for the model's VAE
        width, height = init_image.size
        if width % 8 != 0 or height % 8 != 0:
            print(f"Warning: Image dimensions ({width}x{height}) are not multiples of 8. Resizing to nearest multiple.")
            width = (width // 8) * 8
            height = (height // 8) * 8
            init_image = init_image.resize((width, height))
            # Mask and control_image might also need resizing if used, but are likely ignored here.

        # --- 4. Perform Image Modification (Single Step using Img2Img) ---
        print(f"Performing image modification using Img2Img pipeline...")
        print(f"(Strength: {DEFAULT_STRENGTH}, Scale: {DEFAULT_CFG_SCALE}, Steps: {DEFAULT_NUM_STEPS})")

        # Call the standard Image2Image pipeline.
        # It primarily uses 'prompt', 'image', and 'strength'.
        # Mask/ControlNet related arguments are omitted as they are not standard inputs here.
        modified_image = img2img_pipeline(
            prompt=prompt,
            image=init_image,
            strength=DEFAULT_STRENGTH,
            guidance_scale=DEFAULT_CFG_SCALE,
            num_inference_steps=DEFAULT_NUM_STEPS,
        ).images[0]
        print("Image modification finished.")

        # --- 5. Upload Final Image ---
        final_image_url = save_and_upload_image(modified_image, "final_modified_image")

        # --- 6. Prepare and Return Result ---
        print("\nProcess completed successfully.")
        response = {
            "success": True,
            "final_image_url": final_image_url
            # Add back other details if needed, e.g., prompt
            # "prompt": prompt,
            # "original_image_url": image_url
        }
        return json.dumps(response, indent=2)

    # --- Error Handling ---
    except FileNotFoundError as e:
        error_msg = f"Configuration Error: File or directory not found - {e}. Check model paths like BASE_MODEL_SAVE_DIRECTORY."
        print(f"Error: {error_msg}")
        return json.dumps({"success": False, "error": error_msg})
    except (ConnectionError, ValueError, RuntimeError, Exception) as e:
        # Catch other potential errors during execution
        error_message = f"Diffusion Tool Error: {type(e).__name__} - {e}"
        print(error_message)
        return json.dumps({"success": False, "error": error_message})
    finally:
        # --- Cleanup ---
        # Attempt to release GPU memory
        del controlnet
        del img2img_pipeline
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("Cleaned up models and emptied CUDA cache.")