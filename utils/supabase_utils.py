import os
from supabase import create_client
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

def store_chat_message(user_id, message, response, image_url=None):
    """
    Store a chat message and its response in Supabase.
    
    Args:
        user_id (str): Identifier for the user (can be a session ID if no auth)
        message (str): The user's message
        response (str): The LLM's response
        image_url (str, optional): URL of any image included in the message
        
    Returns:
        dict: The inserted record or None if there was an error
    """
    try:
        timestamp = datetime.now().isoformat()
        
        data = {
            "user_id": user_id,
            "timestamp": timestamp,
            "message": message,
            "response": response,
            "image_url": image_url
        }
        
        result = supabase.table("chat_history").insert(data).execute()
        return result.data[0] if result.data else None
    
    except Exception as e:
        print(f"Error storing chat message: {e}")
        return None

def get_chat_history(user_id, limit=10):
    """
    Retrieve chat history for a specific user.
    
    Args:
        user_id (str): Identifier for the user
        limit (int, optional): Maximum number of messages to retrieve
        
    Returns:
        list: List of chat history records
    """
    try:
        result = supabase.table("chat_history") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("timestamp", desc=True) \
                .limit(limit) \
                .execute()
                
        # Return the messages in chronological order (oldest first)
        return list(reversed(result.data)) if result.data else []
    
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return [] 
    
def get_conversation_history(conversation_id, limit=10):
    """
    Retrieve conversation history for a specific user.
    
    Args:
        conversation_id (str): Identifier for the conversation
        limit (int, optional): Maximum number of messages to retrieve
        
    Returns:
        list: List of chat history records
    """
    try:
        result = supabase.table("prompts") \
                .select("*") \
                .eq("conversation_id", conversation_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
                
        # Return the messages in chronological order (oldest first)
        return list(reversed(result.data)) if result.data else []
    
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return [] 