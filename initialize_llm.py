from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import sys

load_dotenv()

try:
    with open('parameters.json') as f:
        parameters = json.load(f)
    print("[DEBUG] Parameters loaded successfully from parameters.json")
except FileNotFoundError:
    print("Error: parameters.json not found!")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in parameters.json - {e}")
    sys.exit(1)

try:
    main_llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=parameters['decoding']['temperature'],
        top_p=parameters['decoding']['top_p'],
        repetition_penalty=parameters['decoding']['repetition_penalty']
    )

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
    )
    print("[DEBUG] LLM models initialized successfully\n")
except Exception as e:
    print(f"Error: Failed to initialize LLM - {e}")
    print("Make sure you have set GOOGLE_API_KEY in your .env file")
    sys.exit(1)

