import os
import uuid
from flask import Flask, request, jsonify, session
from agent.agent import create_agent_for_user, format_message_with_history

# Cloudinary utilities
from utils.utils import configure_cloudinary, json_parser, upload_image_to_cloudinary
# Supabase utilities
from utils.supabase_utils import store_chat_message, get_chat_history
#Configure Cloudinary
configure_cloudinary()

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects multipart/form-data with:
      - "message": text input (required)
      - "image": optional file (e.g., .jpg or .png)
      - "user_id": optional user identifier (if not provided, will use session ID)
    """
    # Read form data (for multipart/form-data)
    message = request.form.get("message")
    image_file = request.files.get("image")
    user_id = request.form.get("user_id")
    
    # If no user_id is provided, use session ID
    if not user_id:
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        user_id = session['user_id']

    if not message:
        return jsonify({"error": "Missing 'message' parameter"}), 400

    # If there's an image, upload it to Cloudinary
    cloud_url = None
    if image_file:
        try:
            cloud_url = upload_image_to_cloudinary(image_file)
        except Exception as e:
            return jsonify({"error": f"Failed to upload image: {e}"}), 500

    # Get chat history for this user from Supabase
    history = get_chat_history(user_id)
    
    # Format the message with system prompt and history
    formatted_message = message
    if cloud_url:
        formatted_message += f"\nHere is the image URL: {cloud_url}"
        message += f"\nHere is the image URL: {cloud_url}"
    formatted_message = format_message_with_history(formatted_message, history)
    
    # Create a new agent instance for this user
    user_agent = create_agent_for_user()
    
    # Pass the formatted message to the agent
    try:
        result = user_agent.invoke(input=formatted_message)
        
        # Store the message and response in Supabase
        store_chat_message(user_id, message, result['output'], cloud_url)
        
        return jsonify({
            "response": json_parser(result['output']),
            "user_id": user_id
        })
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Get chat history for a user.
    
    Query parameters:
      - user_id: The user identifier (required)
      - limit: Maximum number of messages to retrieve (optional, default=10)
    """
    user_id = request.args.get("user_id")
    limit = request.args.get("limit", default=10, type=int)
    
    if not user_id:
        return jsonify({"error": "Missing 'user_id' parameter"}), 400
    
    history = get_chat_history(user_id, limit)
    
    return jsonify({
        "history": history,
        "user_id": user_id
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)



