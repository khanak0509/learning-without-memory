from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import base_prompt
from evaluation import main as evaluate_response
import json

def test_adaptation():
    """
    Test if LLM is adapting by running the same query multiple times
    and observing how parameters and responses change.
    """
    # Same query for all iterations
    user_input = "Explain quantum computing in 50 words"
    
    print("="*80)
    print("TESTING LLM ADAPTATION")
    print("="*80)
    print(f"Query: {user_input}")
    print(f"Running 3 iterations to observe adaptation...\n")
    
    for iteration in range(1, 4):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}\n")
        
        # Load current parameters
        with open('parameters.json') as f:
            params = json.load(f)
        
        # Create LLM with current parameters
        main_llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            temperature=params['decoding']['temperature'],
            top_p=params['decoding']['top_p'],
            repetition_penalty=params['decoding']['repetition_penalty']
        )
        
        print(f"[BEFORE] Current Parameters:")
        print(f"  Temperature: {params['decoding']['temperature']:.3f}")
        print(f"  Top_p: {params['decoding']['top_p']:.3f}")
        print(f"  Repetition penalty: {params['decoding']['repetition_penalty']:.3f}")
        print(f"  Verbosity: {params['prompt']['verbosity']:.3f}")
        print(f"  w_clarity: {params['weights']['w_clarity']:.3f}\n")
        
        # Generate prompt with current parameters
        prompt_text = base_prompt.format(
            user_input=user_input,
            Verbosity_level=params['prompt']['verbosity'],
            Structure_strictness=params['prompt']['structure_strictness'],
            Creativity_bias=params['prompt']['creativity_bias']
        )
        
        # Get LLM response
        response = main_llm.invoke(prompt_text)
        llm_answer = response.content
        
        print(f"[RESPONSE] LLM Output:")
        print(f"{llm_answer}\n")
        print(f"Word count: {len(llm_answer.split())} words\n")
        
        # Evaluate and update parameters
        print("[EVALUATION] Running evaluation...\n")
        updated_params = evaluate_response(user_input, llm_answer)
        
        print(f"\n[AFTER] Updated Parameters:")
        print(f"  Temperature: {updated_params['decoding']['temperature']:.3f}")
        print(f"  Top_p: {updated_params['decoding']['top_p']:.3f}")
        print(f"  Repetition penalty: {updated_params['decoding']['repetition_penalty']:.3f}")
        print(f"  Verbosity: {updated_params['prompt']['verbosity']:.3f}")
        print(f"  w_clarity: {updated_params['weights']['w_clarity']:.3f}")
        
        # Show the changes
        print(f"\n[CHANGES]:")
        print(f"  Temperature: {params['decoding']['temperature']:.3f} → {updated_params['decoding']['temperature']:.3f} (Δ {updated_params['decoding']['temperature'] - params['decoding']['temperature']:.4f})")
        print(f"  Top_p: {params['decoding']['top_p']:.3f} → {updated_params['decoding']['top_p']:.3f} (Δ {updated_params['decoding']['top_p'] - params['decoding']['top_p']:.4f})")
        print(f"  Repetition penalty: {params['decoding']['repetition_penalty']:.3f} → {updated_params['decoding']['repetition_penalty']:.3f} (Δ {updated_params['decoding']['repetition_penalty'] - params['decoding']['repetition_penalty']:.4f})")
        print(f"  Verbosity: {params['prompt']['verbosity']:.3f} → {updated_params['prompt']['verbosity']:.3f} (Δ {updated_params['prompt']['verbosity'] - params['prompt']['verbosity']:.4f})")
        print(f"  w_clarity: {params['weights']['w_clarity']:.3f} → {updated_params['weights']['w_clarity']:.3f} (Δ {updated_params['weights']['w_clarity'] - params['weights']['w_clarity']:.4f})")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print("="*80)
    print("\nConclusion: Compare the responses and parameter changes across iterations")
    print("to see if the LLM is adapting to produce better outputs.")

if __name__ == "__main__":
    test_adaptation()