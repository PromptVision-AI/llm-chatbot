# PromptVisionAI: An LLM-Powered Image Processing Agent

PromptVisionAI is an intelligent image processing agent that combines the power of Large Language Models (LLMs) with state-of-the-art computer vision models to perform complex visual tasks through natural language instructions.

## 🌟 Features

- **Object Detection** with Grounding DINO
- **Instance Segmentation** with Segment Anything Model (SAM)
- **Image Inpainting/Editing** with Stable Diffusion XL
- **OCR** for text extraction from images
- **Image Captioning** with Florence2 for understanding image content
- **Black & White Conversion** for simple image transformations
- **Context-aware processing** that maintains conversation history
- **Chained operations** automatically triggered by the LLM agent

## 🏗️ Architecture

The agent architecture dynamically chains tools together based on user requests:

![System Architecture](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745639692/image_a4l8ks.png)

The LLM agent serves as the orchestrator, determining which tools to use and in what sequence based on the user's natural language request. This allows for complex workflows to be executed through simple conversational instructions.

## 🛠️ Core Tools

### 1. Object Detection (Grounding DINO)

The agent uses Grounding DINO for zero-shot object detection based on text prompts.

**Model:** Grounding DINO Base
* Architecture: Transformer-based zero-shot detection model
* Framework: HuggingFace Transformers
* Model Path: `tools/grounding_dino/grounding_dino_base`

**Technical Implementation:**
1. **Preprocessing:**
   - Input image is loaded, converted to RGB format
   - Text prompts are formatted as nested lists for model input
   - Inputs are processed using a specialized `AutoProcessor` from Transformers

2. **Detection Process:**
   - Tokenized inputs pass through the model architecture
   - Grounding DINO performs object-text alignment and localization
   - A post-processing step applies threshold filtering (box_threshold=0.45, text_threshold=0.3)
   - Bounding boxes are normalized to the original image dimensions

3. **Postprocessing:**
   - Extracted boxes are converted to [x1, y1, x2, y2] format
   - Centroids are calculated as the center points of each bounding box
   - Confidence scores are rounded to 3 decimal places
   - Matplotlib visualizes boxes on the original image with labels and scores
   - Annotated image is saved as PNG and uploaded to Cloudinary

**Input:**
```json
{
  "image_url": "https://example.com/lion.jpg",
  "prompt": "lion"
}
```

**Output:**
```json
{
  "success": true,
  "prompt": "lion",
  "original_image_url": "https://example.com/lion.jpg",
  "bounding_boxes": [[100, 150, 400, 500]],
  "centroids": [[250, 325]],
  "labels": ["lion"],
  "annotated_image_url": "https://res.cloudinary.com/example/annotated_lion.jpg"
}
```

**Example:**

Original Image:  
![Original Lion](https://res.cloudinary.com/promptvisionai/image/upload/v1745558391/promptvisionai/63f045d5-5c7d-4240-b0cb-ce45faf509bb/inputs/18_input.jpg)

Object Detection (Bounding Box):  
![Detected Lion](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558406/tcloikbmlvpilx6sbu6r.png)

### 2. Instance Segmentation (SAM)

The Segment Anything Model (SAM) creates precise masks for objects detected in the previous step.

**Model:** SAM (Segment Anything Model) v2.1 Large
* Architecture: Mask decoder architecture
* Framework: Ultralytics SAM implementation
* Model Path: `tools/sam/sam2.1_l.pt`

**Technical Implementation:**
1. **Preprocessing:**
   - Input image is downloaded and stored as a temporary file
   - Bounding boxes from Grounding DINO are passed directly to SAM

2. **Segmentation Process:**
   - SAM model inference runs on the image with provided bounding boxes
   - The model generates binary mask predictions for each bounding box

3. **Postprocessing:**
   - Multiple masks (if detected) are merged into a single binary mask using logical OR
   - Advanced morphological operations enhance mask quality:
     * Opening operation (erosion then dilation) removes noise
     * Closing operation (dilation then erosion) fills holes
     * Small isolated regions (<20 pixels) are removed
     * Final dilation expands the mask slightly
     * Gaussian blur with thresholding smooths edges
   - Processed mask is converted to a PNG image and uploaded to Cloudinary

**Input:**
```json
{
  "image_url": "https://example.com/lion.jpg",
  "bounding_boxes": [[100, 150, 400, 500]]
}
```

**Output:**
```json
{
  "success": true,
  "original_image_url": "https://example.com/lion.jpg",
  "merged_mask_url": "https://res.cloudinary.com/example/lion_mask.jpg"
}
```

**Example:**

Segmentation Mask:  
![Segmented Lion](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558413/fu0bergks65emy7rfsw0.png)

### 3. Image Inpainting (Stable Diffusion XL)

The agent uses Stable Diffusion XL for high-quality inpainting to modify objects based on text prompts. The implementation is a two-stage process:

**Models:**
* **Base Model:** Stable Diffusion XL Inpainting
  * Architecture: Latent diffusion model with U-Net backbone
  * Framework: HuggingFace Diffusers
  * Model Path: `tools/diffusion/LatentDiffusion/sdxl_inpaint_base_local`

* **Refiner Model:** Stable Diffusion XL Img2Img
  * Architecture: Latent diffusion model specialized for refinement
  * Framework: HuggingFace Diffusers
  * Model Path: `tools/diffusion/LatentDiffusion/sdxl_refiner_local`

**Technical Implementation:**
1. **Base Inpainting Model (SDXL Inpaint):** 
   - Takes the original image, a mask, and a text prompt
   - Performs initial inpainting up to a defined denoising threshold (0.8)
   - Outputs latent representations rather than a final image
   - Uses 28 inference steps with DPMSolverMultistepScheduler
   - Applies a default negative prompt to avoid unwanted artifacts: "blurry, low quality, distortion, mutation, watermark, signature, text, words"

2. **Refiner Model (SDXL Img2Img):**
   - Takes the latents from the base model
   - Continues the denoising process from where the base model stopped (denoising_start=0.8)
   - Uses 15 inference steps to refine details and add photorealism
   - Outputs the final high-quality image

3. **Memory Optimization:**
   - Implements VRAM management between stages
   - Moves the base model to CPU after first stage
   - Uses model CPU offloading
   - Enables xformers for memory-efficient attention
   - Implements VAE tiling for processing larger images
   - Uses garbage collection and CUDA cache clearing

4. **Preprocessing:**
   - Image dimensions are verified/adjusted to multiples of 8 (required by VAE)
   - Images larger than VAE thresholds are resized
   - Mask is prepared with Gaussian blur (sigmaX=3, sigmaY=3) and thresholding (127)
   - Handles resolution consistency between image and mask using PIL.Image.resize

**Input:**
```json
{
  "image_url": "https://example.com/lion.jpg",
  "mask_url": "https://example.com/lion_mask.jpg",
  "prompt": "orange cat"
}
```

**Output:**
```json
{
  "success": true,
  "inpainted_image_url": "https://res.cloudinary.com/example/lion_to_cat.jpg",
  "original_image_url": "https://example.com/lion.jpg"
}
```

**Example:**

Inpainted Image (Lion to Cat):  
![Lion to Cat](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558467/jyqqbbqsx6yevryn0bpm.png)

### 4. Image Captioning (Florence2)

The image captioning tool uses Florence2 to generate descriptive captions for images, providing context for further processing.

**Model:** Florence2 Vision-Language Model
* Architecture: Multimodal transformer with vision and language capabilities
* Framework: Custom API hosted service
* API Endpoint: External Florence API 

**Technical Implementation:**
1. **Preprocessing:**
   - Input image is downloaded from provided URL
   - Image format is determined and preserved for API request
   - A temporary filename with appropriate extension is generated

2. **API Interaction:**
   - Image is sent to the Florence API endpoint as a multipart/form-data request
   - API does the heavy lifting of running the Florence2 model
   - Response includes the generated detailed caption

3. **Note:**
   - The Florence2 capabilities are made available through a custom API ([Florence API](https://github.com/PromptVision-AI/florence_api))
   - Processing is done on the server-side, reducing client-side compute requirements

**Input:**
```json
{
  "image_url": "https://example.com/lion.jpg"
}
```

**Output:**
```json
{
  "success": true,
  "caption": "A lion lying in the grass with trees in the background on the savanna.",
  "original_image_url": "https://example.com/lion.jpg"
}
```

### 5. OCR (Optical Character Recognition)

The OCR tool extracts text from images, useful for reading signs, documents, or any text content in visual materials.

**Model:** Tesseract OCR
* Architecture: LSTM-based OCR engine
* Framework: pytesseract (Python wrapper for Tesseract)
* Version: Tesseract 4.x

**Technical Implementation:**
1. **Preprocessing:**
   - Input image is downloaded from provided URL
   - Image is loaded using PIL's Image module

2. **Text Extraction:**
   - Pytesseract's `image_to_string` function processes the image
   - No specialized preprocessing is applied to the image
   - Extracted text is stripped of leading/trailing whitespace

3. **Postprocessing:**
   - Simple validation checks if any text was found
   - Returns a message if no text could be extracted

**Input:**
```json
{
  "image_url": "https://example.com/sign.jpg"
}
```

**Output:**
```json
{
  "success": true,
  "extracted_text": "NO PARKING\nVIOLATORS WILL BE TOWED",
  "original_image_url": "https://example.com/sign.jpg"
}
```

### 6. Black & White Conversion

A simple image transformation tool that converts colored images to black and white.

**Technical Implementation:**
1. **Preprocessing:**
   - Input image is downloaded from the provided URL
   - Image is loaded into memory using PIL

2. **Conversion Process:**
   - PIL's `convert("L")` method transforms the image to grayscale
   - This creates an 8-bit single channel image where each pixel has a value from 0 (black) to 255 (white)

3. **Postprocessing:**
   - Converted image is saved to a temporary file as PNG
   - Image is uploaded to Cloudinary for storage
   - Temporary local file is deleted after upload

**Input:**
```json
{
  "image_url": "https://example.com/colorful_image.jpg"
}
```

**Output:**
```json
{
  "success": true,
  "original_image_url": "https://example.com/colorful_image.jpg",
  "bw_image_url": "https://res.cloudinary.com/example/bw_image.jpg"
}
```

## 🔄 Workflow Examples

The LLM agent can dynamically chain different tools together based on the task. Here are some example workflows:

### Example 1: Object Editing (Detection → Segmentation → Inpainting)

1. **User Query**: "Change this lion to orange cat"
2. **LLM Agent**: Recognizes this as an inpainting task requiring detection and segmentation first
3. **Object Detection**: Identifies and localizes lion with bounding box
4. **Segmentation**: Creates precise mask of the lion
5. **Inpainting**: Replaces lion with orange cat based on mask

**Result:**  
![Lion to Cat](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558467/jyqqbbqsx6yevryn0bpm.png)

### Example 2: Contextual Image Processing

The agent maintains context of the conversation and can work with previously modified images:

1. **User Query**: "Change the lion to a horse"
   - Agent performs detection → segmentation → inpainting

   **Result:**  
   ![Lion to Horse](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558677/vtf1akahhjvgth9ikzop.png)

2. **User Query**: "Change this horse to a goat"
   - Agent performs new detection → segmentation → inpainting on the horse image

   **Object Detection:**  
   ![Detected Horse](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745563738/dm58uxl7cnt3xbkcawai.png)

   **Segmentation:**  
   ![Segmented Horse](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745563745/w4k13m9kkhavk6ixbuyp.png)

   **Final Result:**  
   ![Horse to Goat](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745563793/f2tueea22vuvj1bdi7yl.png)

### Example 3: Text Extraction (Captioning → OCR)

1. **User Query**: "What does this sign say?"
2. **LLM Agent**: Recognizes this requires text extraction
3. **Image Captioning**: Identifies that there's text in the image 
4. **OCR**: Extracts the text content from the image
5. **LLM Response**: Provides the extracted text with context

### Example 4: Image Understanding (Captioning → Detection)

1. **User Query**: "Tell me what's in this image"
2. **LLM Agent**: Recognizes this requires image understanding
3. **Image Captioning**: Generates a general description of the image
4. **Object Detection**: Identifies specific objects if needed for more detail
5. **LLM Response**: Combines the caption and detection results to provide a comprehensive description

## 🔧 Technical Implementation

### LLM Agent Architecture

The system uses an LLM (Llama-3.3-70b via Groq) to:
1. Parse user requests
2. Determine the required sequence of operations
3. Call appropriate tools in the correct order
4. Generate natural language responses

The agent architecture consists of:
- **System Prompt**: Guides the LLM's behavior
- **Tool Definitions**: Provide the LLM with capabilities
- **Memory**: Tracks conversation context
- **Execution Engine**: Manages the workflow between tools

### Image Processing Tools

1. **Object Detection (Grounding DINO)**
   - Takes an image URL and text prompt
   - Returns bounding boxes and centroids of detected objects
   - Produces annotated images showing detections

2. **Segmentation (SAM)**
   - Takes an image URL and bounding boxes
   - Returns a binary mask of the segmented object
   - Uses morphological operations to improve mask quality

3. **Inpainting (Stable Diffusion XL)**
   - Takes an image URL, mask URL, and text prompt
   - Uses a two-stage pipeline (Base + Refiner)
   - Returns a modified image with the specified changes

4. **Image Captioning (Florence2)**
   - Takes an image URL
   - Returns a descriptive caption of the image content
   - Powered by a custom Florence2 API ([GitHub](https://github.com/PromptVision-AI/florence_api))

5. **OCR**
   - Takes an image URL
   - Returns extracted text from the image
   - Useful for reading signs, documents, and other text in images

6. **Black & White Conversion**
   - Takes an image URL
   - Returns a black and white version of the image
   - Simple but effective image transformation

## 🚀 Setup and Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU for optimal performance (CPU mode is also supported)
- 10+ GB of disk space for model weights

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PromptVision-AI/llm-chatbot.git
   cd llm-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


3. **Download required model weights:**
   
   - **Grounding DINO**:
     ```bash
     mkdir -p tools/grounding_dino/grounding_dino_base
     # Download model files from huggingface.co/IDEA-Research/grounding-dino-base
     # Place model files in tools/grounding_dino/grounding_dino_base directory
     ```

   - **SAM (Segment Anything Model)**:
     ```bash
     mkdir -p tools/sam
     # Download SAM weights from Facebook Research
     # Place sam2.1_l.pt in the tools/sam directory
     ```

   - **Stable Diffusion XL**:
     ```bash
     mkdir -p tools/diffusion/LatentDiffusion/sdxl_inpaint_base_local
     mkdir -p tools/diffusion/LatentDiffusion/sdxl_refiner_local
     # Download SDXL base inpainting model and refiner model
     # Place in their respective directories
     ```

4. **Configure environment variables:**
   
   Create a `.env` file in the project root with the following required variables:
   ```
   # LLM API (Groq)
   GROQ_API_KEY=your_groq_api_key
   LLM_NAME=llama-3.3-70b-versatile
   
   # Cloudinary for image hosting
   CLOUDINARY_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret
   
   # Supabase for chat history
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

5. **Set up Supabase:**
   - Create a new project in Supabase
   - Run the SQL setup script to create the required tables (see below)

6. **Start the server:**
   ```bash
   python main.py
   ```
   The server will start on port 5000 by default.

## 📝 API Endpoints

### Chat Endpoint

```
POST /chat
```

Parameters (JSON):
- `user_id`: User identifier
- `prompt_id`: Prompt identifier
- `prompt`: User's message/instruction
- `conversation_id`: Conversation identifier
- `input_image_url`: URL for image (optional)

Response:
```json
{
  "text_response": "I've changed the lion to an orange cat.",
  "image_url": "https://res.cloudinary.com/example/original_lion.jpg",
  "inpainted_image_url": "https://res.cloudinary.com/example/lion_to_cat.jpg"
}
```

