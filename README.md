# PromptVisionAI: An LLM-Powered Image Processing Agent

PromptVisionAI is an intelligent image processing agent that combines the power of Large Language Models (LLMs) with state-of-the-art computer vision models to perform complex visual tasks through natural language instructions.

## 🌟 Features

- **Object Detection** with Grounding DINO
- **Instance Segmentation** with Segment Anything Model (SAM)
- **Image Inpainting/Editing** with Stable Diffusion XL
- **OCR** for text extraction from images
- **Image Captioning** for understanding image content
- **Black & White Conversion** for simple image transformations
- **Context-aware processing** that maintains conversation history
- **Chained operations** automatically triggered by the LLM agent

## 📋 Processing Pipeline

The agent follows a systematic approach for handling complex image editing tasks:

1. **Object Detection** → **Segmentation** → **Inpainting**

![Image Processing Pipeline](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558406/tcloikbmlvpilx6sbu6r.png)

## 🛠️ Core Tools

### 1. Object Detection (Grounding DINO)

The agent uses Grounding DINO for zero-shot object detection based on text prompts.

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

The agent uses Stable Diffusion XL for high-quality inpainting to modify objects based on text prompts.

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

## 🔄 Workflow Examples

### Example 1: Change Lion to Orange Cat

1. **User Query**: "Change this lion to orange cat"
2. **LLM Agent**: Recognizes this as an inpainting task requiring detection and segmentation first
3. **Object Detection**: Identifies and localizes lion with bounding box
4. **Segmentation**: Creates precise mask of the lion
5. **Inpainting**: Replaces lion with orange cat based on mask

**Result:**  
![Lion to Cat](https://res.cloudinary.com/dpjbjlvu7/image/upload/v1745558467/jyqqbbqsx6yevryn0bpm.png)

### Example 2: Change Lion to Horse to Goat

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

### Image Processing Pipeline

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

## 🚀 Setup

Please refer to the installation instructions in the original README for details on:
- Dependencies installation
- Model weights download
- Environment variables configuration
- Supabase setup for chat history

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

