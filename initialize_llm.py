from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()

with open('parameters.json') as f:
    params = json.load(f)

main_llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=params['decoding']['temperature'],
    top_p=params['decoding']['top_p'],
    repetition_penalty=params['decoding']['repetition_penalty']
)

eval_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
