from langchain_core.tools import tool  
from utils.utils import json_parser

@tool
def multiply_numbers(input: str) -> int:
    """
    Multiplies two numbers provided in a JSON string.

    The input should be a JSON-formatted string representing a dictionary with:
      - "n1": int, the first number.
      - "n2": int, the second number.

    Returns:
      int: The product of n1 and n2.
    """
    try:
        data = json_parser(input)
        n1 = int(data.get("n1"))
        n2 = int(data.get("n2"))
    except Exception as e:
        raise ValueError(f"Invalid input. Expected JSON with keys 'n1' and 'n2'. Error: {str(e)}")
    print(f"Multiplying numbers {n1} and {n2}")
    return n1 * n2
