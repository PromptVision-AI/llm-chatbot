# LLM Chatbot with Supabase Chat History

A Flask-based chatbot application that uses Groq's LLM API for generating responses, with tools for mathematical operations and image processing. The application includes a system prompt to guide the LLM's behavior and Supabase integration for storing and retrieving chat history.

## Features

- Chat with an LLM (Llama-3.3-70b via Groq)
- System message properly set for the agent
- Per-user agent instances to prevent context mixing
- Conversation memory for maintaining context
- Mathematical operations (sum and multiply numbers)
- Image processing (convert to black and white)
- Image upload via Cloudinary
- Chat history storage and retrieval via Supabase

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   CLOUDINARY_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

4. Set up Supabase:
   - Create a new project in Supabase
   - Create a table called `chat_history` with the following columns:
     - `id` (uuid, primary key)
     - `user_id` (text)
     - `timestamp` (timestamp with timezone)
     - `message` (text)
     - `response` (text)
     - `image_url` (text, nullable)

5. Run the application:
   ```
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

## Customizing the System Prompt

You can modify the system prompt in `utils/prompts.py` to change how the LLM behaves. The system prompt is passed to the agent as a `SystemMessage` when it's initialized, ensuring that it properly guides the model's behavior throughout the conversation.

## Conversation Memory

The application creates a new agent instance with its own memory for each user request. This ensures that conversation contexts don't get mixed between different users. Each agent uses LangChain's `ConversationBufferWindowMemory` to maintain context during a conversation, storing the last 5 interactions. The chat history from Supabase is loaded into this memory when processing a user's request.