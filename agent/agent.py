# LangChain / LLM Imports
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
# Prompt utilities
from agent.prompts import SYSTEM_PROMPT

from agent.config_llm import tools, llm

# Function to create a new agent with its own memory
def create_agent_for_user(user_history=None):
    """
    Create a new agent instance with its own memory for a specific user.
    
    Args:
        user_history (list, optional): List of previous chat messages for this user
        
    Returns:
        agent: A new agent instance with its own memory
    """
    # Create a new memory instance for this user
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,  # Remember last 5 interactions
        return_messages=True
    )
    
    # Load previous messages into memory if available
    if user_history:
        for entry in user_history[-5:]:  # Load last 5 messages
            memory.chat_memory.add_user_message(entry['message'])
            memory.chat_memory.add_ai_message(entry['response'])
    
    # Create a system message
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    
    # Initialize a new agent with its own memory
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        system_message=system_message,
    )
    
    return agent