from langchain_core.tools import tool  
from utils.utils import json_parser
@tool
def sum_numbers(input: str) -> int:
    """
    Adds two numbers provided in a JSON string.

    The input should be a JSON-formatted string representing a dictionary with the following keys:
      - "n1": int, the first number.
      - "n2": int, the second number.

    Returns:
        int: The sum of n1 and n2.
    """
    try:
        data = json_parser(input)
        n1 = int(data.get("n1"))
        n2 = int(data.get("n2"))
    except Exception as e:
        raise ValueError(f"Invalid input. Expected a JSON string with keys 'n1' and 'n2'. Error: {str(e)}")
    print(f"Validating user with numbers {n1} and {n2}")
    return n1 + n2