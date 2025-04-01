# LLM Chatbot with Supabase Chat History

A Flask-based chatbot application that uses Groq's LLM API for generating responses, with tools for mathematical operations and image processing. The application includes a system prompt to guide the LLM's behavior and Supabase integration for storing and retrieving chat history.

## Features

- Chat with an LLM (Llama-3.3-70b via Groq)
- System message properly set for the agent
- Per-user agent instances to prevent context mixing
- Conversation memory for maintaining context
- Mathematical operations (sum and multiply numbers)
- Image processing capabilities:
  - Convert images to black and white
  - Object detection using Grounding DINO
  - Image segmentation using SAM (Segment Anything Model)
- Image upload via Cloudinary
- Chat history storage and retrieval via Supabase

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required model weights:
   - Grounding DINO:
     ```bash
     mkdir -p grounding_dino_base
     # Download the model from: https://huggingface.co/IDEA-Research/grounding-dino-base
     # Place the model files in the grounding_dino_base directory
     ```
   - SAM (Segment Anything Model):
     ```bash
     # Download SAM weights from: https://github.com/facebookresearch/segment-anything
     # Place sam2.1_l.pt in the project root directory
     ```

4. Create a `.env` file with the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   CLOUDINARY_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

5. Set up Supabase:
   - Create a new project in Supabase
   - Run the following SQL in the Supabase SQL editor to create the required table:
     ```sql
     CREATE TABLE chat_history (
         id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
         user_id TEXT NOT NULL,
         timestamp TIMESTAMPTZ NOT NULL,
         message TEXT NOT NULL,
         response TEXT NOT NULL,
         image_url TEXT,
         created_at TIMESTAMPTZ DEFAULT NOW()
     );

     CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
     CREATE INDEX idx_chat_history_timestamp ON chat_history(timestamp);

     ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

     CREATE POLICY "Users can view their own chat history"
         ON chat_history FOR SELECT
         USING (auth.uid()::text = user_id);

     CREATE POLICY "Users can insert their own messages"
         ON chat_history FOR INSERT
         WITH CHECK (auth.uid()::text = user_id);
     ```

6. Run the application:
   ```bash
   python main.py
   ```

## API Endpoints

### Chat Endpoint

```
POST /chat
```

Parameters (multipart/form-data):
- `message` (required): The user's message
- `image` (optional): An image file
- `user_id` (optional): User identifier (if not provided, a session ID will be used)

Response:
```json
{
  "response": "The LLM's response",
  "user_id": "user_identifier"
}
```

### History Endpoint

```
GET /history?user_id=<user_id>&limit=<limit>
```

Parameters:
- `user_id` (required): User identifier
- `limit` (optional, default=10): Maximum number of messages to retrieve

Response:
```json
{
  "history": [
    {
      "id": "uuid",
      "user_id": "user_identifier",
      "timestamp": "2023-01-01T12:00:00Z",
      "message": "User message",
      "response": "LLM response",
      "image_url": "https://cloudinary.com/image.jpg"
    }
  ],
  "user_id": "user_identifier"
}
```

## Model Weights

The following model weights are required but not included in the repository due to size limitations:

1. Grounding DINO Base Model:
   - Download from: https://huggingface.co/IDEA-Research/grounding-dino-base
   - Place in: `grounding_dino_base/` directory
   - Required files:
     - config.json
     - pytorch_model.bin
     - preprocessor_config.json
     - special_tokens_map.json
     - tokenizer_config.json
     - tokenizer.json
     - vocab.txt

2. SAM (Segment Anything Model):
   - Download from: https://github.com/facebookresearch/segment-anything
   - Place `sam2.1_l.pt` in the project root directory

## Customizing the System Prompt

You can modify the system prompt in `agent/prompts.py` to change how the LLM behaves. The system prompt is passed to the agent as a `SystemMessage` when it's initialized, ensuring that it properly guides the model's behavior throughout the conversation.

## Conversation Memory

The application creates a new agent instance with its own memory for each user request. This ensures that conversation contexts don't get mixed between different users. Each agent uses LangChain's `ConversationBufferWindowMemory` to maintain context during a conversation, storing the last 5 interactions. The chat history from Supabase is loaded into this memory when processing a user's request.