# Learning Without Memory

This project explores whether a Large Language Model can adapt its behavior without storing conversation history or relying on traditional training methods.

Rather than remembering previous interactions, the system adjusts generation parameters like temperature, verbosity, and repetition penalty based on how well each response meets the desired outcome.

The learned behavior is stored as a small set of numeric parameters in a JSON file.

## 🔍 What's the Idea?

Traditional learning approaches rely on:
- Conversation memory
- Fine-tuning on datasets
- Embeddings or retrieval systems

This project takes a different approach:

**Can an LLM improve future outputs by updating only its generation parameters, using feedback from the current output?**

Think of it as teaching by adjusting knobs, not by storing examples.

## 🧠 Core Loop

1. **Ask a question**
   - Example: "Explain quantum computing in 50 words."

2. **Generate a response**
   - Uses parameters from `parameters.json`
   - No conversation history is passed

3. **Evaluate the response**
   - Word count accuracy
   - Repetition analysis (rule-based)
   - Clarity score (using a secondary LLM, scored 0 to 1)

4. **Update parameters**
   - Too long → decrease verbosity and temperature
   - Too short → increase verbosity and temperature
   - Repetitive → increase repetition penalty

5. **Save updated parameters**
   - Only numeric values are persisted
   - No outputs or prompts are stored

6. **Next run uses updated behavior**

## ⚙️ Parameters

All learned behavior is stored in `parameters.json`.

### Decoding Parameters (LLM Call)

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `temperature` | 0.1 – 1.2 | Controls randomness |
| `top_p` | 0.7 – 0.99 | Nucleus sampling |
| `repetition_penalty` | 1.0 – 2.0 | Penalizes repeated tokens |

### Prompt Parameters (Injected into Prompt)

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `verbosity` | 0.0 – 1.0 | Controls response length |
| `structure_strictness` | 0.0 – 1.0 | Enforces organization |
| `creativity_bias` | 0.0 – 1.0 | Factual vs creative |

### Evaluation Weights (Controller Only)

| Parameter | Purpose |
|-----------|---------|
| `w_length` | Importance of word-count accuracy |
| `w_clarity` | Importance of clarity score |
| `w_format` | Importance of structure |

## 🧪 Observations

**Test case:** "Explain quantum computing in 50 words."

- **Starting with high verbosity (1.0)**
  - Model already outputs approximately 48 to 52 words and stabilizes

- **Starting with low verbosity (0.1)**
  - Gradual increase across runs
  - Values: 0.10 → 0.146 → 0.20 → 0.32

- **After convergence (verbosity approximately 0.3 to 0.4)**
  - The model consistently produces around 50 words

### Key Result

Once trained, if you ask:

```
"Explain quantum computing"
```
(without specifying length)

The model still outputs approximately 50 words, because that behavior is now encoded in the parameters.

No memory. No training. Just learned behavior.

## ⚠️ Limitations

- Requires multiple runs to converge
- Explicit prompt instructions (such as "exactly 100 words") override learned behavior
- Learning is local and task-specific
- No long-term generalization across different domains (by design)

## 📦 Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

## 📁 Files

- `main.py` - run a single query
- `test.py` - train with 3 iterations
- `test_learned_behavior.py` - test without explicit word limits
- `evaluation.py` - calculate metrics and update parameters
- `llm_config.py` - LLM setup
- `prompt.py` - prompt templates
- `schema_class.py` - output validation
- `parameters.json` - stored parameters

## 🚀 Usage

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

## 📚 References

- [LangChain Python](https://python.langchain.com/)
- [ChatGoogleGenerativeAI](https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/)
