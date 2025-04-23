# LangChain / LLM Imports
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
# Prompt utilities
from agent.prompts import SYSTEM_PROMPT

from agent.config_llm import tools, llm

def format_message_with_history(message, user_history=None):
    """
    Format the input message with system prompt and chat history.
    
    Args:
        message (str): Current user message
        user_history (list, optional): List of previous chat messages for this user
        
    Returns:
        str: Formatted message with system prompt and history
    """
    formatted_message = f"System Prompt:\n\n{SYSTEM_PROMPT}\n\n"
    
    if user_history:
        formatted_message += "History of the conversation:\n\n"
        # Get last 6 interactions
        for entry in user_history[-6:]:
            formatted_message += f"User: {entry['text']}\n" ## UPDATE WITH text
            formatted_message += f"Assistant: {entry['response']}\n\n"
    
    formatted_message += f"Current user message:\n\n{message}"
    
    return formatted_message

# Function to create a new agent
def create_agent_for_user(user_history=None):
    """
    Create a new agent instance for a specific user.
    
    Args:
        user_history (list, optional): List of previous chat messages for this user
        
    Returns:
        agent: A new agent instance
    """
    # Initialize a new agent with OPENAI_FUNCTIONS type which can handle both tool and non-tool responses
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Changed from ZERO_SHOT_REACT_DESCRIPTION
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=15  # Limit the number of iterations to prevent infinite loops
    )
    
    return agent