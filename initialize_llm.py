from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json 
load_dotenv()

with open('parameters.json') as f:
    parameters = json.load(f)

llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = parameters['decoding']['temperature'],
    top_p = parameters['decoding']['top_p'],
    repetition_penalty=parameters['decoding']['repetition_penalty']
)
