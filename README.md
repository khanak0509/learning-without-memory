# Learning Without Memory

A simple experiment to see if an LLM can learn to adjust its behavior by updating generation parameters instead of storing conversation history.

## What's This About?

Instead of remembering previous conversations, this system adjusts parameters like temperature and verbosity based on how well the LLM's responses match what you asked for. It's like teaching the model by tweaking knobs rather than showing it examples.

The idea is simple: 
- Run a query and get a response
- Check if the response was good (right length, clear, not repetitive)
- Adjust the parameters based on what went wrong
- Save those parameters for next time

No memory needed, just learned behavior stored in a few numbers.

## How It Works

1. **Ask a question** - e.g., "Explain quantum computing in 50 words"
2. **LLM responds** using current parameters from `parameters.json`
3. **Evaluate the response:**
   - Count words (did it match the target?)
   - Check for repetition (how many words repeated?)
   - Ask another LLM to rate clarity (0-1 score)
4. **Update parameters:**
   - If too long → decrease verbosity and temperature
   - If too short → increase verbosity and temperature
   - If repetitive → increase repetition penalty
5. **Save to JSON** for next time

The learning rate is 0.1, so changes happen gradually over multiple runs.

## Parameters

The system tracks these in `parameters.json`:

**Decoding:**
- `temperature` (0.1-1.2) - randomness
- `top_p` (0.7-0.99) - sampling threshold
- `repetition_penalty` (1.0-2.0) - how much to avoid repeating words

**Prompt:**
- `verbosity` (0.0-1.0) - controls response length
- `structure_strictness` (0.0-1.0) - how organized the response should be
- `creativity_bias` (0.0-1.0) - factual vs creative

**Weights:**
- `w_length`, `w_clarity`, etc. - how much each metric matters

## What I Found

After testing with "Explain quantum computing in 50 words":

- Starting from high verbosity (1.0): Already produces ~48 words, stays there
- Starting from low verbosity (0.1): Gradually increases (0.1 → 0.146 → 0.200) as it learns the output is too short
- After training to verbosity ~0.3-0.4, the model learns the behavior

**The cool part:** Once trained, if you just ask "Explain quantum computing" (without specifying word count), it automatically gives around 50 words because it learned that pattern.

So yeah, it does learn something. The parameters encode behavior without needing to remember past conversations.

## Limitations

- Takes multiple runs to converge
- When you give an explicit word count in the prompt, that overrides the parameter

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

## Files

- `main.py` - run a single query
- `test.py` - train with 3 iterations
- `test_learned_behavior.py` - test without explicit word limits
- `evaluation.py` - calculate metrics and update parameters
- `llm_config.py` - LLM setup
- `prompt.py` - prompt templates
- `schema_class.py` - output validation
- `parameters.json` - stored parameters

## Usage

Train the system:
```bash
python test.py
```

Test learned behavior:
```bash
python test_learned_behavior.py
```

Single run:
```bash
python main.py
```

## References

- [LangChain Python](https://python.langchain.com/)
- [ChatGoogleGenerativeAI](https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/)
