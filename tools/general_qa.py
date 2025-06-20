from langchain_core.tools import tool  
from utils.utils import json_parser
import json

@tool
def general_qa_tool(input: str) -> str:
    """
    Use this tool for answering general questions that are not related to other tools.
    This tool should be used when the question is about general knowledge, explanations, or discussions
    that don't require any specific tool functionality

    input: The input should be the user query 

    Returns:
      The user query
    """
    
    return json.dumps({"response": f"Reflect about the following and provide a response: {input}"})