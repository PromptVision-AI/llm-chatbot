import os
import requests
from io import BytesIO
import json
import torch
import numpy as np
import cv2 # Needed for mask blurring
from PIL import Image
# import matplotlib # Not needed
# matplotlib.use('Agg') # Not needed
import tempfile
import shutil # For potential cleanup on error
import warnings

from diffusers import (
    StableDiffusionXLInpaintPipeline,   # For Base
    StableDiffusionXLImg2ImgPipeline,   # For Refiner
    DPMSolverMultistepScheduler         # Scheduler
    # ControlNetModel removed
)
from langchain_core.tools import tool

# Assuming utility functions are available from your project structure
# Make sure these imports work in your actual project setup
from utils.utils import json_parser, upload_image_to_cloudinary


# --- Configuration Constants ---
# Define Local Load Paths (Relative to script execution location)
LATENT_DIFFUSION_FOLDER = "X:/PromptVision/llm-chatbot/tools/diffusion/LatentDiffusion"
BASE_MODEL_LOAD_PATH = os.path.join(LATENT_DIFFUSION_FOLDER, "sdxl_inpaint_base_local")
REFINER_MODEL_LOAD_PATH = os.path.join(LATENT_DIFFUSION_FOLDER, "sdxl_refiner_local")

# Default generation parameters for SDXL Base + Refiner Inpainting
DEFAULT_BASE_STEPS = 28
DEFAULT_REFINER_STEPS = 15
DEFAULT_CFG_SCALE = 8.0         # Guidance scale
DEFAULT_DENOISING_END = 0.8     # Where base stops and refiner starts
DEFAULT_NEG_PROMPT = ("blurry, low quality, distortion, mutation, watermark, signature, text, words") # Default negative prompt

DTYPE = torch.float16 # Use float16 for efficiency

# --- Helper Functions ---
def download_image(url):
    """Downloads an image from a URL and returns a PIL Image."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Downloaded image from {url}")
        return image
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download image from {url}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to process image data from {url}: {e}")

# Removed make_inpaint_condition

def prepare_mask(mask_image_pil: Image.Image) -> Image.Image:
    """Applies blur and thresholding to the mask for smoother blending."""
    try:
        if not isinstance(mask_image_pil, Image.Image):
             raise TypeError("Input must be a PIL Image object.")
        # Ensure mask is grayscale before processing
        mask_l = mask_image_pil.convert("L")
        mask_np = np.array(mask_l)
        # Blur slightly
        mask_blurred_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=3, sigmaY=3)
        # Threshold back to binary
        mask_binary_np = (mask_blurred_np > 127).astype(np.uint8) * 255
        prepared_mask = Image.fromarray(mask_binary_np, mode="L")
        print("Prepared mask (blurred and thresholded).")
        return prepared_mask
    except Exception as e:
        raise ValueError(f"Failed to prepare mask: {e}")


def save_and_upload_image(image: Image.Image, step_name="image"):
    """Saves a PIL image temporarily, uploads it via helper, returns URL, cleans up."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, format="PNG")
            print(f"Saved intermediate image for upload: {temp_path}")

        print(f"Attempting to upload {step_name} from {temp_path}")
        image_url = upload_image_to_cloudinary(temp_path) # Assumes this function exists
        print(f"Upload successful for {step_name}: {image_url}")
        return image_url
    except Exception as e:
        print(f"Error during save/upload for {step_name}: {e}")
        raise ValueError(f"Failed to save or upload {step_name}: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {cleanup_e}")


# --- Tool Definition ---
@tool
def diffusion_inpainting_tool(input: str) -> str:
    """
    Performs high-quality SDXL inpainting using a Base + Refiner workflow.
    Loads models from local directories ('./LatentDiffusion/sdxl_inpaint_base_local' and
    './LatentDiffusion/sdxl_refiner_local'). Takes an initial image, a mask specifying the
    area to change, and a text prompt. Resizes images if dimensions are not multiples of 8.
    Uploads the final refined image to Cloudinary.

    The input should be a JSON-formatted string with:
      - "image_url": str, URL of the initial image.
      - "mask_url": str, URL of the mask image (white areas indicate where to inpaint).
      - "prompt": str, text description of what to generate in the masked area.

    Returns:
      str: A JSON-formatted string containing:
        - "success": bool, True if completed successfully, False otherwise.
        - "final_image_url": str, Cloudinary URL of the final generated image (if successful).
        - "error": str, (Optional) Error message if success is False.
    """
    # Explicitly trigger garbage collection and cache clearing
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleaned up CUDA cache.")   
    # --- Verify Configuration ---
    # ... (configuration check remains the same) ...
    if not os.path.exists(BASE_MODEL_LOAD_PATH):
        error_msg = f"Configuration Error: Local base model path not found: {BASE_MODEL_LOAD_PATH}"
        print(f"Error: {error_msg}")
        return json.dumps({"success": False, "error": error_msg})
    if not os.path.exists(REFINER_MODEL_LOAD_PATH):
        error_msg = f"Configuration Error: Local refiner model path not found: {REFINER_MODEL_LOAD_PATH}"
        print(f"Error: {error_msg}")
        return json.dumps({"success": False, "error": error_msg})


    base_pipeline = None # Initialize for cleanup
    refiner_pipeline = None # Initialize for cleanup
    try:
        # --- 1. Parse Input ---
        # ... (input parsing remains the same) ...
        print("Parsing input JSON...")
        data = json_parser(input) # Assumes this helper exists
        image_url = data.get("image_url")
        mask_url = data.get("mask_url")
        prompt = data.get("prompt")

        if not all([image_url, mask_url, prompt]):
            raise ValueError("Missing required input: 'image_url', 'mask_url', or 'prompt'.")
        print("Input parsed successfully.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ... (device check remains the same) ...

        # --- 2. Load Models Locally ---
        # ... (model loading remains the same) ...
        print(f"Loading base pipeline from local path: {BASE_MODEL_LOAD_PATH}")
        base_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            BASE_MODEL_LOAD_PATH, torch_dtype=DTYPE
        )
        print(f"Loading refiner pipeline from local path: {REFINER_MODEL_LOAD_PATH}")
        refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            REFINER_MODEL_LOAD_PATH, torch_dtype=DTYPE
        )

        # --- 3. Apply Optimizations & Runtime Settings ---
        # ... (optimizations remain the same) ...
        print("Applying optimizations and runtime settings...")
        pipelines_to_optimize = [base_pipeline, refiner_pipeline]
        # ... (loop to apply optimizations) ...
        for i, pipe in enumerate(pipelines_to_optimize):
            pipe_name = "Base" if i == 0 else "Refiner"
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.safety_checker = None # Disable safety checker
            pipe.enable_model_cpu_offload() # Use CPU offload for memory saving
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print(f"  - Enabled xformers for {pipe_name}")
            except ImportError: print(f"  - xformers not installed for {pipe_name}.")
            except Exception as e: print(f"  - Could not enable xformers for {pipe_name}: {e}")
            pipe.enable_vae_tiling()
            print(f"  - Applied settings to {pipe_name}")
        print("Pipelines loaded and configured.")


        # --- 4. Prepare Images ---
        print("Downloading and preparing images...")
        init_image_pil = download_image(image_url)
        mask_image_pil = download_image(mask_url)

        # <<< START: New Resizing Logic >>>
        original_width, original_height = init_image_pil.size
        print(f"Original image size: {original_width}x{original_height}")

        # Check if dimensions are multiples of 8 (required by VAE)
        if original_width % 8 != 0 or original_height % 8 != 0:
            # Calculate new dimensions by rounding DOWN to the nearest multiple of 8
            new_width = (original_width // 8) * 8
            new_height = (original_height // 8) * 8

            # Ensure dimensions are not zero after rounding down
            if new_width == 0 or new_height == 0:
                 error_msg = f"Image dimensions ({original_width}x{original_height}) are too small and resulted in zero size after rounding down to multiple of 8."
                 raise ValueError(error_msg)

            print(f"Warning: Image dimensions are not multiples of 8. Resizing to nearest multiple: {new_width}x{new_height}")
            # Resize both image and mask to the new calculated dimensions
            init_image = init_image_pil.resize((new_width, new_height), Image.LANCZOS)
            # Check if mask size matches original image size before resizing mask
            if mask_image_pil.size == (original_width, original_height):
                 mask_image = mask_image_pil.resize((new_width, new_height), Image.NEAREST)
            else:
                 # If mask size didn't match image initially, resize it based on its own original size
                 # (or handle this case as an error depending on requirements)
                 print(f"Warning: Initial mask size {mask_image_pil.size} differs from image size {original_width}x{original_height}. Resizing mask independently.")
                 m_w, m_h = mask_image_pil.size
                 nm_w = (m_w // 8) * 8
                 nm_h = (m_h // 8) * 8
                 if nm_w == 0 or nm_h == 0:
                      raise ValueError(f"Mask dimensions ({m_w}x{m_h}) too small.")
                 # We still need the mask to match the resized init_image size for the pipeline
                 # So, resize the mask to the *image's* new dimensions anyway
                 mask_image = mask_image_pil.resize((new_width, new_height), Image.NEAREST)


        else:
            # Dimensions are already multiples of 8, no resize needed for VAE compatibility
            print(f"Using original image size: {original_width}x{original_height} (already multiple of 8)")
            init_image = init_image_pil
            # Ensure mask matches image size if no resize happened
            if mask_image_pil.size != init_image.size:
                 print(f"Warning: Mask size {mask_image_pil.size} differs from image size {init_image.size}. Resizing mask.")
                 mask_image = mask_image_pil.resize(init_image.size, Image.NEAREST)
            else:
                 mask_image = mask_image_pil
        # <<< END: New Resizing Logic >>>

        # Prepare mask (blur/threshold) using the potentially resized mask_image
        prepared_mask = prepare_mask(mask_image) # mask_image is now the correctly sized version

        # --- 5. Setup Generator ---
        # ... (generator setup remains the same) ...
        generator = torch.Generator(device=device) #.manual_seed(42)

        # --- 6. Run Base Inpainting Stage ---
        # ... (base stage remains the same, uses resized 'init_image' and 'prepared_mask') ...
        print(f"Running base inpainting stage ({DEFAULT_BASE_STEPS} steps)...")
        latents = base_pipeline(
            prompt=prompt,
            negative_prompt=DEFAULT_NEG_PROMPT,
            image=init_image, # Use potentially resized image
            mask_image=prepared_mask, # Use potentially resized & prepared mask
            num_inference_steps=DEFAULT_BASE_STEPS,
            guidance_scale=DEFAULT_CFG_SCALE,
            denoising_end=DEFAULT_DENOISING_END,
            generator=generator,
            output_type="latent",
        ).images
        print("Base stage finished.")


        # --- 7. VRAM Management ---
        # ... (VRAM management remains the same) ...
        print("Moving base pipeline to CPU, clearing cache...")
        # ... (base_pipeline.to("cpu"), empty_cache, latents.cpu()) ...
        base_pipeline.to("cpu") # Move base off GPU
        del base_pipeline # Explicitly delete to help GC
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        latents = latents.cpu() # Move latents to CPU
        print("VRAM cleared.")


        # --- 8. Run Refiner Stage ---
        # ... (refiner stage remains the same) ...
        print(f"Running refiner stage ({DEFAULT_REFINER_STEPS} steps)...")
        latents = latents.to(device, dtype=DTYPE) # Move latents back to device
        final_image = refiner_pipeline(
            prompt=prompt,
            negative_prompt=DEFAULT_NEG_PROMPT,
            image=latents,
            denoising_start=DEFAULT_DENOISING_END,
            num_inference_steps=DEFAULT_REFINER_STEPS,
            guidance_scale=DEFAULT_CFG_SCALE,
            generator=generator,
        ).images[0]
        print("Refiner stage finished.")


        # --- 9. Upload Final Image ---
        # ... (upload remains the same) ...
        final_image_url = save_and_upload_image(final_image, "final_refined_inpainted_image")


        # --- 10. Prepare and Return Result ---
        # ... (result formatting remains the same) ...
        print("\nProcess completed successfully.")
        response = {
            "success": True,
            "inpainted_image_url": final_image_url,
            "original_image_url": image_url,
        }
        return json.dumps(response)

    # --- Error Handling ---
    # ... (error handling remains the same) ...
    except FileNotFoundError as e:
         error_msg = f"Configuration Error: File or directory not found - {e}. Check model paths."
         print(f"Error: {error_msg}")
         return json.dumps({"success": False, "error": error_msg})
    except (ConnectionError, ValueError, RuntimeError, ImportError, TypeError, Exception) as e:
        error_message = f"Diffusion Tool Error: {type(e).__name__} - {e}"
        print(error_message)
        if isinstance(e, (ValueError, TypeError)):
             import traceback
             print(traceback.format_exc())
        return json.dumps({"success": False, "error": error_message})
    finally:
        # --- Cleanup ---
        # ... (cleanup remains the same) ...
        # Delete local variables pointing to pipelines/models
        # Check if variables exist before deleting, in case of early error
        if 'base_pipeline' in locals() and base_pipeline is not None: del base_pipeline
        if 'refiner_pipeline' in locals() and refiner_pipeline is not None: del refiner_pipeline
        # Explicitly trigger garbage collection and cache clearing
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("Cleaned up pipelines and emptied CUDA cache.")