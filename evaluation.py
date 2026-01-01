import re 
import json
from numpy import clip 

with open("parameters.json") as f:
    parameters = json.load(f)

alpha = 0.1 

def extract_target_words(user_input):
    match = re.search(r'(\d+)\s*words?', user_input.lower())
    if match:
        return int(match.group(1))
    return None

# print(extract_target_words("sumarise in 100 word"))

def count_words(llm_ans):
    words = len(llm_ans.strip().split())
    return words

def main(user_input , llm_ans):
    actual_words = count_words(llm_ans)
    target_words = extract_target_words(user_input)
    error  = (actual_words - target_words) / target_words
    parameters["prompt"]["verbosity"] = clip(parameters["prompt"]["verbosity"] - alpha * error, 0.0, 1.0)
    parameters["decoding"]["temperature"] = clip(parameters["decoding"]["temperature"] - 0.5 * alpha * error, 0.1, 1.2)
    parameters["weights"]["w_length"] = clip(parameters["weights"]["w_length"] + alpha * abs(error), 0.5, 5.0)
    




    