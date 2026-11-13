# Part 2: Local AI Writing Assistant with LangChain

> **Series:** [Part 1: Ollama](https://github.com/rizwannazir333/ollama-email-english) · **Part 2: LangChain** · [Part 3: LangGraph](https://github.com/rizwannazir333/langgraph-email-english)

Build an **email writer** and an **English grammar checker** that run **100% on your own computer**.
No API key, no internet, no cost.

In [Part 1](https://github.com/rizwannazir333/ollama-email-english) we built this app with plain Ollama.
Now we rebuild it with **LangChain**, a library of **ready-made building blocks** that snap together
like LEGO. You will see how LangChain adds memory, tools, structured answers and more with very little code.

**What you will learn: 10 LangChain features**

| # | Feature | What it does in this project |
|---|---|---|
| 1 | `ChatOllama` | The local model, wrapped so LangChain can use it |
| 2 | Prompt templates | Prompts with blanks like `{tone}` and `{today}` |
| 3 | Few-shot examples | Worked corrections the model learns from |
| 4 | Chains | `prompt \| model \| parser`: data flows left to right |
| 5 | Structured output | The correction comes back as a **Python object**, not just text |
| 6 | Memory | Remembers the last email, so *"make it shorter"* works |
| 7 | Batch | Corrects many sentences with one call |
| 8 | Retry + fallback | If the model fails: try again, then use a backup model |
| 9 | Callbacks | Code that runs *while* the model works (we print the thinking) |
| 10 | Tools + agent | The model calls Python functions, e.g. to find the date of "next Friday" |

> **Teacher's tip:** open [`2_langchain_email_english.py`](2_langchain_email_english.py) and search for
> `FEATURE`. Every feature is marked right where it happens in the code.

---

## 1. What you need

- A computer with **8 GB RAM or more** (16 GB recommended). **No graphics card needed.**
- **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/))
- **Ollama**, free, runs AI models on your computer ([ollama.com/download](https://ollama.com/download))
- About **4 GB of free disk space** for the model

---

## 2. Setup (10 minutes)

**Step 1: download the AI model.**

```bash
ollama pull qwen3.5:4b
```

**Step 2: get the code.**

```bash
git clone https://github.com/rizwannazir333/langchain-email-english.git
cd langchain-email-english
```

**Step 3: create a virtual environment and install the libraries.**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Step 4: run it.**

```bash
python 2_langchain_email_english.py
```

```
[langchain | model: qwen3.5:4b | thinking: off]

1) Draft email          2) Revise last email
3) Correct English      4) Correct many lines
5) Email agent (finds real dates)      q) Quit
>
```

> New to local AI? The key words (LLM, quantized model, prompt, token, thinking) are explained in
> [Part 1](https://github.com/rizwannazir333/ollama-email-english#3-key-ideas).

---

## 3. The lesson: LangChain step by step

### Chains: the heart of LangChain

The `|` symbol (a "pipe") connects steps. The output of one step becomes the input of the next:

```python
email_chain   = email_prompt   | make_model(model, think) | StrOutputParser()
correct_chain = (correct_prompt
                 | make_model(model, think, Correction.model_json_schema())
                 | PydanticOutputParser(pydantic_object=Correction))
```

Read it as: **fill in the prompt, then ask the model, then turn the answer into what we need.**

### Prompt templates: prompts with blanks

```python
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write short, polite emails in a {tone} tone. Today is {today}. " + RULES),
    MessagesPlaceholder("history"),     # earlier messages go here (memory)
    ("human", "{request}"),
])
```

We write the prompt **once** and fill in `{tone}`, `{today}` and `{request}` every time.

### Few-shot examples: teach by showing

```python
FewShotChatMessagePromptTemplate(
    examples=[example_1, example_2],
    example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
)
```

The model sees two finished corrections (*"She goed to the market..."*, *"Me and him was late."*)
before your sentence, and copies their style.

### Structured output: answers your code can use

We describe the shape we want with **Pydantic**:

```python
class Change(BaseModel):
    old: str
    new: str
    reason: str

class Correction(BaseModel):
    corrected: str
    changes: list[Change]
```

The model is forced to answer in this JSON shape, and `PydanticOutputParser` turns it into a Python
object. Now `result.corrected` is a string and `result.changes` is a list. No messy text parsing!

### Memory: the model remembers the conversation

```python
history = []   # earlier messages

def draft_email(chains, request, tone="professional"):
    answer = chains["email"].invoke({"request": request, "tone": tone,
                                     "history": history, ...})
    history.extend([HumanMessage(request), AIMessage(answer)])   # remember this turn
    return answer
```

Draft an email (option 1), then choose option 2 and type *"make it shorter"*. The model sees the old
email in `history` and rewrites it.

### Batch: many inputs, one call

```python
results = chains["correct"].batch([{"text": line} for line in lines], ...)
```

Option 4 corrects many lines at once with the **same chain**.

### Retry and fallback: don't crash

```python
main.with_retry(stop_after_attempt=2).with_fallbacks([backup])
```

If the main model fails, LangChain tries again, and then switches to a backup model.

### Callbacks: run code while the model works

```python
class ShowThinking(BaseCallbackHandler):
    def on_llm_new_token(self, token, chunk=None, **kwargs):
        ...  # print the thinking text as it arrives
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[tool] {serialized['name']} {input_str}")
```

LangChain calls these methods automatically on every event. That's how we show thinking and tool calls.

### Tools and agents: the model asks your code for help

AI models don't know today's date. So we give them a **tool**, an ordinary Python function:

```python
@tool
def next_weekday(day: str) -> str:
    """Return the date of the next given weekday, e.g. 'friday'."""
    ...
```

An **agent** is a model that decides *by itself* when to call a tool:

```python
agent = create_agent(model, tools=[get_today, next_weekday], system_prompt="...")
```

Try option 5 with *"ask my manager for next Friday off"*:

```
[tool] next_weekday {'day': 'friday'}
[tool result] Friday 20 November 2026
===== RESULT =====
Subject: Request for Leave - Family Event
...I would like to request time off next Friday, November 20th...
```

> **Teacher's tip:** the docstring (`"""Return the date..."""`) is not just a comment. The model
> **reads it** to decide when to use the tool. Write clear docstrings!

### What LangChain *can't* do easily

LangChain is great for **straight lines** (A → B → C). But what if the AI should **review its own work
and try again**, or **pause and ask you** before finishing?
👉 That is what **[Part 3: LangGraph](https://github.com/rizwannazir333/langgraph-email-english)** is for.

---

## 4. Command options

```bash
python 2_langchain_email_english.py                                     # menu
python 2_langchain_email_english.py email "ask my manager for Friday off"
python 2_langchain_email_english.py --tone friendly email "invite the team to lunch on Thursday"
python 2_langchain_email_english.py correct "they was going to the shop and buyed some bread"
python 2_langchain_email_english.py agent "ask my manager for next Friday off"
python 2_langchain_email_english.py --think correct "..."               # show thinking (slower)
```

| Option | Meaning |
|---|---|
| `--tone TONE` | Email tone: `friendly`, `formal`, `apologetic`... |
| `--think` | Show the model's thinking live (more accurate, **much slower**) |
| `-m MODEL` | Use another Ollama model, e.g. `-m llama3.2:3b` |

If you don't choose a model, the script uses the **first installed one** from this list:
`qwen3.5:4b` → `llama3.2:3b` → `qwen2.5:3b` → `qwen3.5:latest`. The backup model is `qwen2.5:3b`.

---

## 5. Speed

Tested on a laptop with **no graphics card** (Intel i7, 16 GB RAM): **about 20–45 seconds per answer**.
The script keeps answers short, uses a small context (`num_ctx=2048`), keeps the model loaded
(`keep_alive="30m"`), and has **thinking off by default**, because thinking can take minutes on a CPU.

> **Honest note:** small models make mistakes. They sometimes add a sentence you didn't ask for, or
> write *"I and he"* instead of *"He and I"*. **Always read an AI email before you send it.**

---

## 6. Exercises

1. Add a third few-shot example, e.g. *"I has two cat."* → *"I have two cats."*
2. Add a field `tone: str` to the `Correction` model that describes the tone of the text.
3. Add a new tool `count_words(text)` and tell the agent to keep emails under 80 words.
4. Make option 2 (revise) work for corrections too: *"explain the second change"*.
5. Change `BACKUP_MODEL` to a model you **don't** have. What happens when the main model fails?

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `model "qwen3.5:4b" not found` | Run `ollama pull qwen3.5:4b` |
| `Connection refused` | Start the Ollama app (or run `ollama serve`) |
| `ModuleNotFoundError: No module named 'langchain'` | Do [Setup step 3](#2-setup-10-minutes) |
| The menu seems to do nothing | Type the **number** first, press Enter, *then* your text |
| `Sorry, something went wrong: ...` | The small model returned bad JSON. Just try again |
| Very slow (minutes) | Don't use `--think`, run one script at a time, check `ollama ps` |

---

## Files

```
.
├── 2_langchain_email_english.py   # the app
├── requirements.txt               # Python libraries (tested versions)
├── .gitignore                     # keeps .venv and cache files out of Git
└── README.md                      # this lesson
```

**Previous:** [Part 1: Ollama](https://github.com/rizwannazir333/ollama-email-english) ·
**Next:** [Part 3: the same app with LangGraph](https://github.com/rizwannazir333/langgraph-email-english)
