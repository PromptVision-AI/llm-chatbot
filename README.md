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

![System Architecture](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745650188/IP_Architecture.drawio_t4op4u.png)

The LLM agent serves as the orchestrator, determining which tools to use and in what sequence based on the user's natural language request. This allows for complex workflows to be executed through simple conversational instructions.

## 🛠️ Core Tools

### 1. Object Detection (Grounding DINO)

The agent uses Grounding DINO for zero-shot object detection based on text prompts.

**Model:** Grounding DINO Base
* Architecture: Transformer-based zero-shot detection model
* Framework: HuggingFace Transformers
* Model Path: `tools/grounding_dino/grounding_dino_base`

**Model Description**  
Grounding DINO Base is a moderately heavy model (~1.2 GB) that offers excellent accuracy for zero-shot object-detection tasks. It leverages pretrained weights from the DINO (DIstillation with NO labels) framework, trained on a mixture of Object365, GoldG and COCO datasets. The model excels at locating objects described by natural-language prompts without category-specific training, achieving **56.7 % AP on COCO validation data**.

**Performance Metrics**

- **mAP (COCO val2017)**: **56.7 %** [[Grounding DINO v0.1.0-alpha2 release]](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha2)

**Why We Selected This Model**  
Grounding DINO was chosen over detectors such as DETR, YOLOv8 or Faster R-CNN because of its ability to perform zero-shot detection from text alone. Our system must answer arbitrary user requests, so a fixed-class model like YOLOv8 (80 categories) would require continual re-training. Grounding DINO can interpret and locate virtually any object described in plain language with state-of-the-art accuracy, giving us the flexibility we need while keeping fine-tuning requirements to a minimum.

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

**Model Description**  
SAM v2.1 Large (~2.5 GB) is Meta AI’s flagship promptable segmentation model, pretrained on the 1 B-mask SA-1B dataset (11 M images). It delivers state-of-the-art mask quality and excels at prompt-guided, zero-shot segmentation without task-specific finetuning, reaching **79.5 % mIoU** on high-quality COCO-style benchmarks.

**Performance Metrics**

| Metric | Value | Dataset / Setting | Source |
|--------|-------|-------------------|--------|
| **Boundary AP** | **28.2 AP<sub>B</sub>** | COCO val (ViT-B backbone) | [“Segment Anything in High Quality”](https://ar5iv.org/html/2306.01567) |
| **mIoU** | **79.5 %** | Four HQ datasets (zero-shot, ViT-L) | [“Segment Anything in High Quality”](https://ar5iv.org/html/2306.01567) |
| **Zero-shot mIoU** | **70.6 %** | Four HQ datasets (ViT-B) | [“Segment Anything in High Quality”](https://ar5iv.org/html/2306.01567) |


**Why We Selected This Model**  
SAM outperformed alternatives like Florence 2 in both boundary quality (28.2 AP<sub>B</sub>) and overall mask accuracy (79.5 mIoU). Its prompt-guided zero-shot segmentation removes the need for retraining when users ask to segment new objects, a flexibility other models lack. This high-quality, “segment-anything” capability is essential for our pipeline, where precise masks feed directly into downstream in-painting editing tasks.

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

### Stable Diffusion XL Inpainting Pipeline

**Model Description**  
The SDXL inpainting pipeline is resource-intensive, with the Base + Refiner checkpoints occupying ≈ 10 GB of disk and needing 10 GB + of VRAM for best results. Both networks inherit the pretrained weights from Stability AI’s SDXL-1.0 (trained on billions of LAION-5B image–text pairs and several curated high-quality subsets). The two-stage design first produces a coarse image (Base) and then sharpens details and global lighting (Refiner), yielding photorealistic, seamless edits that rival closed-source systems.

**Performance Metrics**


* **FID** ≈ 23.5 (lower is better, indicates high visual quality) [[MLCommons blog, Aug 28 2024]](https://mlcommons.org/2024/08/sdxl-mlperf-text-to-image-generation-benchmark/) |
* **CLIP Score** ≈ 31.75 (higher is better, measures text-image alignment) [[MLCommons blog, Aug 28 2024]](https://mlcommons.org/2024/08/sdxl-mlperf-text-to-image-generation-benchmark/) |

**Why We Selected This Model**  
We benchmarked several open-source inpainting solutions (e.g., Stable Diffusion 1.5 Inpaint, Kandinsky 2.2, Paint-by-Example) and found SDXL’s two-stage approach consistently produced more coherent lighting and texture transitions while maintaining strong prompt fidelity. Although SDXL demands more VRAM than single-stage models, its near-state-of-the-art FID and CLIP scores—and, more importantly, visibly cleaner boundaries—made it the best fit for our editing use-case.

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

**Model Description**  
Florence 2 Large (≈ 0.77 B parameters) is Microsoft’s state-of-the-art vision–language foundation model. We deploy it behind an internal API, so it introduces no local GPU overhead. Pre-trained on the 5-billion-annotation FLD-5B corpus and then optionally fine-tuned on public captioning data, Florence 2 excels at producing detailed, context-rich descriptions of images.

**Performance Metrics**


 * **CIDEr** = **135.6**  (COCO Karpathy test split, *zero-shot*) [[Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks]](https://arxiv.org/pdf/2311.06242)
 * **CIDEr** = **143.3**  (COCO Karpathy test split, *fine-tuned generalist model*) [[Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks]](https://arxiv.org/pdf/2311.06242)

**Why We Selected This Model**  
In our internal benchmarks Florence 2 produced more accurate, attribute-aware captions than other alternatives, while using fewer parameters than most other models. Its high CIDEr scores translate into captions that faithfully capture subtle details—crucial for downstream decision-making in our pipeline.

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

**Model Description**  
Tesseract OCR v4.x is Google-maintained, LSTM-based open-source software that weighs in at only ≈ 30 MB. Trained on a mixture of public-domain texts and synthetic data, it delivers reliable recognition on clean, printed documents while supporting 100 + languages out of the box.

**Performance Metrics**

| Metric | Value | Evaluation Setup | Source |
|--------|-------|------------------|--------|
| **Character-level accuracy** | **95 – 98 %** | Clean, printed text | [ML Journey – “TrOCR vs. Tesseract” (2024-11-23)](https://mljourney.com/trocr-vs-tesseract-comparison-of-ocr-tools-for-modern-applications/) |
| **Word-level accuracy** | **94 – 98 %** | UNLV dataset, clean docs | [GdPicture Blog – “Best C# OCR libraries: 2025 Guide” (2025-03-25)](https://www.gdpicture.com/blog/best-csharp-ocr-libraries/) |
| **Language support** | **100 + languages** | Official README | [tesseract-ocr/tesseract GitHub README](https://github.com/tesseract-ocr/tesseract) |


**Why We Selected This Model**  
Tesseract’s 95 – 98 % character accuracy on clean documents, broad language coverage, and zero licensing cost made it the best fit for our signage-and-label OCR component. Its tight integration with `pytesseract` lets us embed recognition in our pipeline with minimal dependencies, and its small footprint keeps resource usage low compared to heavier Transformer-based OCR models.

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

PromptVisionAI employs a function-calling agent architecture powered by Llama-3.3-70b (accessed via Groq's API). This high-parameter model serves as the central orchestrator, providing both the reasoning capabilities and the user interface for the entire system.

#### Agent Configuration

The agent follows a structured function-calling pattern where:

1. **Initialization**: On startup, the agent loads tool definitions, system prompts, and establishes connections to external services (Cloudinary for image hosting and Supabase for persistence).

2. **Execution Model**: The agent operates in a predict-execute-observe loop:
   - **Predict**: The LLM determines which tools to call based on user input
   - **Execute**: The system runs the selected tools with specified parameters
   - **Observe**: Results are collected and fed back to the LLM for next steps

3. **Routing Logic**: All user messages are passed through the primary LLM interface, which dynamically decides whether to handle the request directly or delegate to specialized tools.

#### Chain of Thought Reasoning

The system leverages the LLM's inherent chain-of-thought capabilities to:

1. **Parse Intent**: Extract the user's primary goal from natural language
2. **Decompose Tasks**: Break complex operations into sequential tool executions
3. **Plan Execution**: Determine which tools must be used in which order
4. **Interpret Results**: Evaluate tool outputs to decide on next steps
5. **Generate Responses**: Convert technical results into user-friendly language

For example, when a user requests "Change this lion to an orange cat," the agent internally reasons: "I need to (1) detect the lion, (2) create a precise mask, (3) use inpainting to replace it with a cat, and (4) return the result with an explanation."

#### System Prompt Engineering

The system prompt is carefully designed with several components:

1. **Role Definition**: Establishes the agent as an image processing assistant
2. **Tool Documentation**: Detailed descriptions of each tool's capabilities and limitations
3. **Decision Guidelines**: Rules for determining which tools to use for which tasks
4. **Response Formatting**: Instructions on how to present results to users
5. **Error Handling**: Procedures for gracefully managing failures

The prompt instructs the agent to think step-by-step, consider context from prior conversation turns, and prioritize user intent over literal command interpretation.

#### Tool Selection Algorithm

The agent's tool selection process follows these principles:

1. **Context Analysis**: Examines both the immediate request and conversation history
2. **Task Classification**: Categorizes the request into high-level tasks (detection, editing, etc.)
3. **Tool Matching**: Maps tasks to appropriate tools based on capability alignment
4. **Dependency Resolution**: Identifies prerequisites (e.g., detection before segmentation)
5. **Parameter Extraction**: Determines required inputs for each tool from the user request

This process allows the agent to automatically build complex workflows like the detection → segmentation → inpainting chain demonstrated in the examples.

#### Conversation Memory

The system implements a persistent memory architecture:

1. **Storage Backend**: Supabase database maintains conversation history with table schemas optimized for quick retrieval
2. **Context Window**: Recent conversation turns are included in each LLM prompt
3. **Image References**: Prior image URLs are tracked to enable chained operations
4. **Tool Call History**: Previous tool executions are recorded to inform future decisions


This memory system enables the contextual awareness demonstrated in Example 2, where the agent recognizes that "this horse" refers to the previously modified image.

#### Execution Engine

The execution engine coordinates the workflow between tools:

1. **Resource Management**: GPU memory is carefully allocated between different models
2. **Error Recovery**: Failed operations trigger fallback strategies when possible
3. **Result Caching**: Processed images and intermediate results are cached to avoid redundant computation
4. **State Tracking**: The execution state is monitored to ensure proper tool sequencing

### Image Processing Tools

The system integrates six specialized tools (detailed in previous sections):

1. **Object Detection** (Grounding DINO): Zero-shot object localization from text prompts
2. **Segmentation** (SAM): Precise mask generation for detected objects
3. **Inpainting** (Stable Diffusion XL): Two-stage image editing with base and refiner models
4. **Image Captioning** (Florence2): Generates descriptive text from image content
5. **OCR**: Extracts text from images using Tesseract
6. **Black & White Conversion**: Simple grayscale transformation

The agent dynamically chains these tools together based on task requirements, creating a flexible system capable of handling a wide range of image processing requests through natural language instructions.

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

