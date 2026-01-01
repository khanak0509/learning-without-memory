import re 
import json
from numpy import clip 
from collections import Counter
from prompt import * 
from initialize_llm import *
from schema_class import * 



with open("parameters.json") as f:
    parameters = json.load(f)

alpha = 0.1 

def extract_target_words(user_input):
    match = re.search(r'(\d+)\s*words?', user_input.lower())
    if match:
        return int(match.group(1))
    return None


def count_words(llm_ans):
    words = len(llm_ans.strip().split())
    return words


def calculate_repetition_score(output):
    words = [w.lower() for w in output.split()]
    if len(words) == 0:
        return 0.0
    counter = Counter(words)
    repeated_words = sum([count-1 for count in counter.values() if count > 1])
    total_words = len(words)
    return min(1.0, repeated_words / total_words) if total_words > 0 else 0.0


def main(user_input , llm_ans):
    print("\n[DEBUG] Starting evaluation...")
    print(f"[DEBUG] Initial parameters:")
    print(f"  - Temperature: {parameters['decoding']['temperature']:.3f}")
    print(f"  - Top_p: {parameters['decoding']['top_p']:.3f}")
    print(f"  - Repetition penalty: {parameters['decoding']['repetition_penalty']:.3f}")
    print(f"  - Verbosity: {parameters['prompt']['verbosity']:.3f}")
    print(f"  - w_clarity: {parameters['weights']['w_clarity']:.3f}")
    
    actual_words = count_words(llm_ans)
    target_words = extract_target_words(user_input)
    
    if target_words is None or target_words == 0:
        length_error = 0
    else:
        length_error = (actual_words - target_words) / target_words
    parameters["prompt"]["verbosity"] = clip(parameters["prompt"]["verbosity"] - alpha * length_error, 0.0, 1.0)
    parameters["decoding"]["temperature"] = clip(parameters["decoding"]["temperature"] - 0.5 * alpha * length_error, 0.1, 1.2)
    parameters["weights"]["w_length"] = clip(parameters["weights"]["w_length"] + alpha * abs(length_error), 0.5, 5.0)

    repetition_score = calculate_repetition_score(llm_ans)

    repetition_error = repetition_score - 0.5 
    parameters["decoding"]["repetition_penalty"] += alpha * repetition_error
    parameters["decoding"]["repetition_penalty"] = min(max(parameters["decoding"]["repetition_penalty"], 1.0), 2.0)
    parameters["decoding"]["top_p"] -= alpha * repetition_error
    parameters["decoding"]["top_p"] = min(max(parameters["decoding"]["top_p"], 0.7), 0.99)

    llm_Clarity_prompt = llm.with_structured_output(Schema)
    chain = Clarity_prompt | llm_Clarity_prompt
    result = chain.invoke({
        'llm_ans' : llm_ans
    })
    clarity_score = result.score
    clarity_error = 0.5 - clarity_score
    parameters["weights"]["w_clarity"] += alpha * clarity_error
    parameters["weights"]["w_clarity"] = min(max(parameters["weights"]["w_clarity"], 0.5), 5.0)
    
    print(f"\n[DEBUG] Evaluation metrics:")
    print(f"  - Length error: {length_error:.3f} (actual: {actual_words}, target: {target_words})")
    print(f"  - Repetition score: {repetition_score:.3f}")
    print(f"  - Clarity score: {clarity_score:.3f}")
    
    print(f"\n[DEBUG] Updated parameters:")
    print(f"  - Temperature: {parameters['decoding']['temperature']:.3f}")
    print(f"  - Top_p: {parameters['decoding']['top_p']:.3f}")
    print(f"  - Repetition penalty: {parameters['decoding']['repetition_penalty']:.3f}")
    print(f"  - Verbosity: {parameters['prompt']['verbosity']:.3f}")
    print(f"  - w_clarity: {parameters['weights']['w_clarity']:.3f}")
    print("[DEBUG] Parameters saved to parameters.json\n")
    
    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)
    
    return parameters
















    