from initialize_llm import main_llm
from prompt import base_prompt
import json

def test_without_limit():
    with open('parameters.json') as f:
        params = json.load(f)
    
    print("Testing learned behavior")
    print(f"Verbosity: {params['prompt']['verbosity']:.3f}\n")
    
    test_queries = [
        "Explain quantum computing",
        "What is artificial intelligence",
        "Describe neural networks"
    ]
    
    for query in test_queries:
        prompt_text = base_prompt.format(
            user_input=query,
            Verbosity_level=params['prompt']['verbosity'],
            Structure_strictness=params['prompt']['structure_strictness'],
            Creativity_bias=params['prompt']['creativity_bias']
        )
        
        response = main_llm.invoke(prompt_text)
        llm_answer = response.content
        word_count = len(llm_answer.split())
        
        print(f"{query}")
        print(f"Response ({word_count} words): {llm_answer}\n")

if __name__ == "__main__":
    test_without_limit()
