# Learning Without Memory

This project explores whether a Large Language Model can adapt its behavior without storing conversation history or relying on traditional training methods.

Rather than remembering previous interactions, the system adjusts generation parameters like temperature, verbosity, and repetition penalty based on how well each response meets the desired outcome.

The learned behavior is stored as a small set of numeric parameters in a JSON file.

## What's the Idea?

Traditional learning approaches rely on conversation memory, fine-tuning on datasets, or embeddings and retrieval systems.

This project takes a different approach: Can an LLM improve future outputs by updating only its generation parameters, using feedback from the current output?

Think of it as teaching by adjusting knobs, not by storing examples.

## How it Works

The system follows a simple loop:

First, you ask a question (for example, "Explain quantum computing in 50 words").

The LLM generates a response using parameters from parameters.json. No conversation history is passed.

Then the system evaluates the response by checking word count accuracy, analyzing repetition, and getting a clarity score from a secondary LLM (scored 0 to 1).

Based on the evaluation, it updates the parameters. If the response was too long, it decreases verbosity and temperature. If too short, it increases them. If repetitive, it increases the repetition penalty.

The updated parameters get saved to the JSON file. Only numeric values are stored, no outputs or prompts.

The next run uses these updated parameters, so the behavior gradually improves.

## Parameters

All learned behavior is stored in parameters.json.

The system tracks three types of parameters:

**Decoding Parameters** (used when calling the LLM):
- temperature (range 0.1 to 1.2) controls randomness
- top_p (range 0.7 to 0.99) for nucleus sampling
- repetition_penalty (range 1.0 to 2.0) penalizes repeated tokens

**Prompt Parameters** (injected into the prompt):
- verbosity (range 0.0 to 1.0) controls response length
- structure_strictness (range 0.0 to 1.0) enforces organization
- creativity_bias (range 0.0 to 1.0) balances factual vs creative responses

**Evaluation Weights** (used by the controller):
- w_length determines importance of word-count accuracy
- w_clarity determines importance of clarity score
- w_format determines importance of structure

## What I Found

I tested this with "Explain quantum computing in 50 words."

When starting with high verbosity (1.0), the model already outputs about 48 to 52 words and stays stable.

When starting with low verbosity (0.1), there's gradual increase across runs. The verbosity goes from 0.10 to 0.146 to 0.20 to 0.32.

After convergence (verbosity around 0.3 to 0.4), the model consistently produces about 50 words.

The interesting part is that once trained, if you just ask "Explain quantum computing" without specifying length, the model still outputs about 50 words. That behavior is now encoded in the parameters.

No memory. No training. Just learned behavior.

## Limitations

The system requires multiple runs to converge. Explicit prompt instructions like "exactly 100 words" will override the learned behavior. The learning is local and task-specific, with no long-term generalization across different domains (this is by design).

## Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then create a .env file and add your API key:

```
GOOGLE_API_KEY=your_key_here
```

## Files

The project includes these files:

main.py runs a single query

test.py trains the system with 3 iterations

test_learned_behavior.py tests without explicit word limits

evaluation.py calculates metrics and updates parameters

llm_config.py handles LLM setup

prompt.py contains prompt templates

schema_class.py defines output validation

parameters.json stores the learned parameters

## Usage

To train the system, run:

```bash
python test.py
```

To test the learned behavior:

```bash
python test_learned_behavior.py
```

For a single run:

```bash
python main.py
```

## References

LangChain Python documentation: https://python.langchain.com/

ChatGoogleGenerativeAI reference: https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/
