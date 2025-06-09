from langchain_core.tools import tool  
from utils.utils import json_parser
import json

@tool
def end_tool(input: str) -> str:
    """
    Use this tool ONLY when ypu have found the answer to the user query and you want to provide a final response
    
    input: The input should be the output of the previous tool

    Returns:
      response: a query that makes the llm reflect about the user query and provide a final response
    """
    response = f"now that you have the answer to the user query, provide a final response to the user using this information: {input}"
    
    return json.dumps({"response": response})