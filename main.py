import os
import uuid
from flask import Flask, request, jsonify, session
from agent.agent import create_agent_for_user, format_message_with_history

# Cloudinary utilities
from utils.utils import configure_cloudinary, format_endpoint_response
# Supabase utilities
from utils.supabase_utils import get_conversation_history, store_chat_message, get_chat_history
#Configure Cloudinary
configure_cloudinary()

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects JSON data with:
      - "user_id": user identifier (required)
      - "prompt_id": prompt identifier (required)
      - "prompt": text input (required)
      - "conversation_id": conversation identifier (required)
      - "input_image_url": optional text URL for image
    """
    # Read JSON data from request
    data = request.json
    
    # Extract parameters
    user_id = data.get("user_id")
    prompt_id = data.get("prompt_id")
    prompt = data.get("prompt")
    conversation_id = data.get("conversation_id")
    input_image_url = data.get("input_image_url")
    
    # Validate required parameters
    if not user_id:
        return jsonify({"error": "Missing 'user_id' parameter"}), 400
    if not prompt_id:
        return jsonify({"error": "Missing 'prompt_id' parameter"}), 400
    if not prompt:
        return jsonify({"error": "Missing 'prompt' parameter"}), 400
    if not conversation_id:
        return jsonify({"error": "Missing 'conversation_id' parameter"}), 400

    # Get chat history for this user from Supabase
    history = get_conversation_history(conversation_id) ## UPDATE WITH get_conversation_history
    
    # Format the message with system prompt and history
    formatted_message = prompt
    if input_image_url:
        formatted_message += f"\nHere is the image URL: {input_image_url}"
        prompt += f"\nHere is the image URL: {input_image_url}"
    formatted_message = format_message_with_history(formatted_message, history)
    
    # Create a new agent instance for this user
    user_agent = create_agent_for_user()
    
    # Pass the formatted message to the agent
    try:
        result = user_agent.invoke(input=formatted_message)
        
        # Store the message and response in Supabase
        # store_chat_message(user_id, prompt, result['output'], input_image_url) ## UPDATE WITH DELETE
        result_data = format_endpoint_response(result, user_id, prompt_id)
        return jsonify(result_data)
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


