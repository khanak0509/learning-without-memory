from initialize_llm import main_llm
from prompt import base_prompt
from evaluation import main as evaluate_response
import json

with open('parameters.json') as f:
    params = json.load(f)

def run_example():
    user_input = "Explain quantum computing in 50 words"
    print(f"[DEBUG] User input: {user_input}")
    
    prompt_text = base_prompt.format(
        user_input=user_input,
        Verbosity_level=params['prompt']['verbosity'],
        Structure_strictness=params['prompt']['structure_strictness'],
        Creativity_bias=params['prompt']['creativity_bias']
    )
    
    response = main_llm.invoke(prompt_text)
    llm_answer = response.content
    print(f"\n[DEBUG] LLM Response:\n{llm_answer}\n")
    
    updated_params = evaluate_response(user_input, llm_answer)
    
    return llm_answer, updated_params

if __name__ == "__main__":
    run_example()