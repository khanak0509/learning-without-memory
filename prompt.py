from langchain_core.prompts import PromptTemplate

base_prompt = PromptTemplate(
    input_variables= ['user_input',"Verbosity_level","Structure_strictness","Creativity_bias"],
    template= """
You are an assistant.

Constraints:
- Verbosity level: {Verbosity_level}
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

