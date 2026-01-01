import re 
import json 

with open "parameters.json" as f:

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
    length_error = (actual_words - target_words) / target_words


    