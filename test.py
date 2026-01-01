from initialize_llm import main_llm
from prompt import base_prompt
from evaluation import main as evaluate_response
import json

def test_adaptation():
    user_input = "Explain quantum computing in 50 words"
    
    print("Testing LLM Adaptation")
    print(f"Query: {user_input}\n")
    
    for iteration in range(1, 4):
        print(f"Iteration {iteration}")
        
        with open('parameters.json') as f:
            params = json.load(f)
        
        prompt_text = base_prompt.format(
            user_input=user_input,
            Verbosity_level=params['prompt']['verbosity'],
            Structure_strictness=params['prompt']['structure_strictness'],
            Creativity_bias=params['prompt']['creativity_bias']
        )
        
        response = main_llm.invoke(prompt_text)
        llm_answer = response.content
        
        print(f"Response: {llm_answer}")
        print(f"Words: {len(llm_answer.split())}")
        
        updated_params = evaluate_response(user_input, llm_answer)
        print(f"Verbosity: {params['prompt']['verbosity']:.3f} -> {updated_params['prompt']['verbosity']:.3f}\n")

if __name__ == "__main__":
    test_adaptation()