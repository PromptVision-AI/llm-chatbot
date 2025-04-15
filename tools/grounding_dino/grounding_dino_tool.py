from langchain_core.tools import tool
from utils.utils import json_parser, upload_image_to_cloudinary
from PIL import Image
from urllib.request import urlopen
import io
import json
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import os

# Initialize model and processor (do this once when module is imported)
device = "cuda" if torch.cuda.is_available() else "cpu"
local_model_path = "tools/grounding_dino/grounding_dino_base"
processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForZeroShotObjectDetection.from_pretrained(local_model_path).to(device)

@tool
def detect_objects_tool(input: str) -> str:
    """
    Detect objects in an image based on a text prompt using Grounding DINO.
    
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
          - "annotated_image_url": str, the url of the annotated image.
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
    
    # Open the image using Pillow
    try:
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image data: {e}")
    
    # Process the image with Grounding DINO
    try:
        # Prepare text prompt
        text_labels = [[prompt]]
        
        # Process image and get predictions
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        
        result = results[0]
        
        # Extract bounding boxes, scores, and labels
        boxes = []
        scores = []
        labels = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            boxes.append(box)
            scores.append(round(score.item(), 3))
            labels.append(label)
        
        # Calculate centroids for each bounding box
        centroids = []
        for box in boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centroids.append([round(cx, 2), round(cy, 2)])
        
        # Create annotated image with bounding boxes
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Draw bounding boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            # Draw the bounding box
            rect = patches.Rectangle(
                (box[0], box[1]),  # (x, y)
                box[2] - box[0],   # width
                box[3] - box[1],   # height
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
            # Add label and confidence score
            ax.text(
                box[0],
                box[1] - 10,
                f"{label} ({score})",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5)
            )
        
        ax.axis("off")
        
        # Save annotated image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
            plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # Upload annotated image to Cloudinary
        try:
            annotated_image_url = upload_image_to_cloudinary(temp_path)
        except Exception as e:
            raise ValueError(f"Failed to upload annotated image to Cloudinary: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Prepare response
        response = {
            "success": True,
            "prompt": prompt,
            "original_image_url": image_url,
            "bounding_boxes": boxes,
            "centroids": centroids,
            "labels": labels,
            "annotated_image_url": annotated_image_url
        }
        
    except Exception as e:
        raise ValueError(f"Failed to process image with Grounding DINO: {e}")
    
    return json.dumps(response) 