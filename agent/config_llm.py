import os

# tool imports
from tools.end_tool import end_tool
from tools.convert_to_bw import convert_to_bw
from tools.grounding_dino.grounding_dino_tool import detect_objects_tool
from tools.general_qa import general_qa_tool
from tools.sam.sam_segmentation import sam_segment_tool
from tools.florence import caption_image_tool
from tools.diffusion.diffusion_tool import diffusion_inpainting_tool
from tools.ocr.ocr import ocr_image
from langchain_groq import ChatGroq
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

#Retrieve the Groq API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#Retrieve the llm name from .env
LLM_NAME = os.getenv("LLM_NAME")
#Bind your tools to the agent
tools = [convert_to_bw, detect_objects_tool, general_qa_tool, sam_segment_tool, caption_image_tool, ocr_image,diffusion_inpainting_tool,end_tool]

#Initialize the LLM 
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=LLM_NAME,
    temperature=0,
)