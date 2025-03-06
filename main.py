import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# LangChain / LLM Imports
from langchain.agents import initialize_agent
from langchain_groq import ChatGroq

# Your tool imports
from tools.sum_numbers import sum_numbers
from tools.multiply_numbers import multiply_numbers
from tools.convert_to_bw import convert_to_bw

# Cloudinary utilities
from utils.utils import configure_cloudinary, upload_image_to_cloudinary

# 1. Load environment variables
load_dotenv()

# 2. Configure Cloudinary
configure_cloudinary()

# 3. Retrieve the Groq API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 4. Bind your tools to the agent
tools = [sum_numbers, multiply_numbers, convert_to_bw]

# 5. Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# 6. Create the Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects multipart/form-data with:
      - "message": text input (required)
      - "image": optional file (e.g., .jpg or .png)
    """
    # Read form data (for multipart/form-data)
    message = request.form.get("message")
    image_file = request.files.get("image")

    if not message:
        return jsonify({"error": "Missing 'message' parameter"}), 400

    # If there's an image, upload it to Cloudinary
    cloud_url = None
    if image_file:
        try:
            cloud_url = upload_image_to_cloudinary(image_file)
        except Exception as e:
            return jsonify({"error": f"Failed to upload image: {e}"}), 500

    # Optionally include the image URL in the user prompt
    if cloud_url:
        message += f"\nHere is the image URL: {cloud_url}"

    # Pass the message (plus optional image reference) to the agent
    try:
        result = agent.run(message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)



