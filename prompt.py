from langchain_core.prompts import PromptTemplate

base_prompt = PromptTemplate(
    input_variables= ['user_input',"Verbosity_level","Structure_strictness","Creativity_bias"],
    template= """
You are an assistant. Provide a response that matches the specified constraints.

Response Style Constraints (0.0 to 1.0 scale):
- Verbosity: {Verbosity_level}
  → Lower values (0.0-0.4): Very brief, concise, to the point
  → Medium values (0.4-0.6): Balanced, moderately detailed
  → Higher values (0.6-1.0): Comprehensive, detailed explanation

- Structure strictness: {Structure_strictness}
- Creativity bias: {Creativity_bias}

Task:
{user_input}
"""
)

Clarity_prompt = PromptTemplate(
    input_variables= ['llm_ans'],
    template="""
You are an evaluator.  
Rate the following text on clarity from 0 to 1, where 1 means very clear and 0 means very unclear.  
Do not provide any text, only a single numeric value between 0 and 1.

Text:
{llm_ans}
"""
)

