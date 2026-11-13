"""
LangChain Writing Assistant: Email Drafting, Revision & Grammar Correction with Tool-Using Agents
   (Part 2 of 3: LangChain)
   Drafts professional emails and corrects English grammar.
   Runs 100% on your computer with a 4-bit quantized model. No external API.

Quick by default. Add --think to see the model's THINKING live (grey);
the RESULT is always printed in one piece at the end.

LangChain features used here (search for "FEATURE"):
    1. ChatOllama          - the local model (thinking on/off)
    2. Prompt templates    - prompts with {variables}
    3. Few-shot examples   - show the model a worked example
    4. Chains              - prompt | model | parser
    5. Structured output   - the answer is a Python object (Pydantic)
    6. Memory              - chat history, so you can revise the last email
    7. Batch               - correct many lines in one call
    8. Retry + fallback    - try again, then use a backup model
    9. Callbacks           - print the thinking while the model works
   10. Tools + agent       - the agent looks up real dates for your email

Run:
    python langchain_writing_assistant.py                          # menu
    python langchain_writing_assistant.py email "ask boss for Friday off"
    python langchain_writing_assistant.py correct "he go to school yesterday"
    python langchain_writing_assistant.py agent "ask boss for next Friday off"
    python langchain_writing_assistant.py --think correct "..."     # show thinking (slower)
"""

import os
import subprocess
import sys

# If started with another Python, re-run with the .venv next to this file
# (or one folder up), so plain "python this_file.py" always works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
for _VENV in (os.path.join(_HERE, ".venv"), os.path.join(os.path.dirname(_HERE), ".venv")):
    _VENV_PY = os.path.join(_VENV, *_BIN)
    if __name__ == "__main__" and os.path.exists(_VENV_PY):
        if os.path.normcase(sys.prefix) != os.path.normcase(_VENV):
            sys.exit(subprocess.call([_VENV_PY, *sys.argv]))
        break

import argparse
import re
import time
from datetime import date, timedelta

import ollama
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

# All 4-bit quantized (Q4_K_M). The first one you have installed is used.
MODELS = ["qwen3.5:4b", "llama3.2:3b", "qwen2.5:3b", "qwen3.5:latest"]
BACKUP_MODEL = "qwen2.5:3b"

os.system("")  # enable colours in the Windows console
GREY, BOLD, RESET = "\033[90m", "\033[1m", "\033[0m"


# FEATURE 9: Callbacks - LangChain calls these while the model works.
class ShowThinking(BaseCallbackHandler):
    thinking = False

    def on_llm_new_token(self, token, chunk=None, **kwargs):
        thought = chunk.message.additional_kwargs.get("reasoning_content") if chunk else None
        if thought:
            if not self.thinking:
                print(f"{GREY}----- thinking -----")
                self.thinking = True
            print(thought, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        if self.thinking:
            print(f"\n----- thinking done -----{RESET}")
            self.thinking = False

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"{GREY}[tool] {serialized['name']} {input_str}{RESET}")

    def on_tool_end(self, output, **kwargs):
        print(f"{GREY}[tool result] {getattr(output, 'content', output)}{RESET}")


CONFIG = {"callbacks": [ShowThinking()]}


# FEATURE 10: Tools - normal Python functions the agent may call.
@tool
def get_today() -> str:
    """Return today's weekday and date."""
    return date.today().strftime("%A %d %B %Y")


@tool
def next_weekday(day: str) -> str:
    """Return the date of the next given weekday, e.g. 'friday'."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if day.lower() not in days:
        return f"Unknown weekday: {day}"
    ahead = (days.index(day.lower()) - date.today().weekday() - 1) % 7 + 1
    return (date.today() + timedelta(days=ahead)).strftime("%A %d %B %Y")


# FEATURE 5: Structured output - the shape the correction must have.
class Change(BaseModel):
    old: str
    new: str
    reason: str


class Correction(BaseModel):
    corrected: str
    changes: list[Change]


# FEATURE 2: Prompt templates
RULES = (
    "Output only a 'Subject:' line, then the email with greeting and sign-off. Keep "
    "the body to 2-4 short sentences. Use ONLY the facts the user gives. Do not add promises, reasons or "
    "details (no lines like 'I will finish my work first' or 'I will be available by email'). "
    "Use [placeholders] for unknown names."
)
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write short, polite emails in a {tone} tone. Today is {today}. " + RULES),
    MessagesPlaceholder("history"),             # FEATURE 6: earlier messages go here
    ("human", "{request}"),
])

# Guardrail: small models like to add promises the user never asked for, such as
# "I will ensure my tasks are completed". Remove those sentences in plain Python
# (instant, no extra model call) - unless the user's own request mentioned them.
PROMISES = ("i will ensure", "i'll ensure", "i will make sure", "i'll make sure",
            "will be available", "remain available", "hand over", "handover")


def remove_unrequested_promises(email: str, request: str) -> str:
    banned = [p for p in PROMISES if p not in request.lower()]
    lines = []
    for line in email.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [s for s in sentences if not any(p in s.lower() for p in banned)]
        if kept or not line.strip():          # drop lines that became empty
            lines.append(" ".join(kept))
    return "\n".join(lines)


# FEATURE 3: Few-shot - worked examples the model can copy.
example_1 = {
    "input": "She goed to the market and buyed two apple.",
    "output": Correction(corrected="She went to the market and bought two apples.", changes=[
        Change(old="goed", new="went", reason="irregular past tense"),
        Change(old="buyed", new="bought", reason="irregular past tense"),
        Change(old="apple", new="apples", reason="plural after 'two'"),
    ]).model_dump_json(),
}
example_2 = {
    "input": "Me and him was late.",
    "output": Correction(corrected="He and I were late.", changes=[
        Change(old="Me and him", new="He and I", reason="subject pronouns"),
        Change(old="was", new="were", reason="plural subject"),
    ]).model_dump_json(),
}
correct_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful English proofreader. Fix grammar, spelling, tense and "
               "word choice. Keep the meaning. List every change; do not list words "
               "that stay the same. Answer in JSON."),
    FewShotChatMessagePromptTemplate(
        examples=[example_1, example_2],
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
    ),
    ("human", "{text}"),
])


# FEATURE 1 + 8: the local model, with retry and a backup model
def make_model(model: str, think: bool, json_schema=None):
    main = ChatOllama(model=model, reasoning=think, format=json_schema, temperature=0.3,
                      num_ctx=4096 if think else 2048,       # smaller = faster
                      num_predict=2500 if think else 600, keep_alive="30m")
    backup = ChatOllama(model=BACKUP_MODEL, format=json_schema, temperature=0.3, num_ctx=2048)
    return main.with_retry(stop_after_attempt=2).with_fallbacks([backup])


def build(model: str, think: bool) -> dict:
    """FEATURE 4: chains - each one is prompt | model | parser."""
    email_chain = email_prompt | make_model(model, think) | StrOutputParser()
    correct_chain = (correct_prompt
                     | make_model(model, think, Correction.model_json_schema())
                     | PydanticOutputParser(pydantic_object=Correction))
    agent = create_agent(
        ChatOllama(model=model, reasoning=think, temperature=0.3, keep_alive="30m",
                   num_ctx=4096 if think else 2048),
        tools=[get_today, next_weekday],
        system_prompt="You write short, polite emails. "
                      "If the request names a weekday (like 'Friday'), FIRST call next_weekday with that day and put the returned date in the email. " + RULES,
    )
    return {"email": email_chain, "correct": correct_chain, "agent": agent}


history = []  # FEATURE 6: memory - the email conversation so far


def draft_email(chains, request: str, tone: str = "professional") -> str:
    inputs = {"request": request, "tone": tone, "history": history,
              "today": date.today().strftime("%A %d %B %Y")}
    answer = remove_unrequested_promises(chains["email"].invoke(inputs, CONFIG), request)
    history.extend([HumanMessage(request), AIMessage(answer)])
    return answer


def correct_text(chains, text: str) -> str:
    c = chains["correct"].invoke({"text": text}, CONFIG)
    changes = [f"- {x.old} -> {x.new} ({x.reason})" for x in c.changes if x.old != x.new] or ["- none"]
    return f"Corrected: {c.corrected}\nChanges:\n" + "\n".join(changes)


def correct_many(chains, lines: list[str]) -> str:
    # FEATURE 7: batch - the same chain on many inputs (one after another on CPU)
    results = chains["correct"].batch([{"text": line} for line in lines],
                                      {**CONFIG, "max_concurrency": 1}, return_exceptions=True)
    out = []
    for line, c in zip(lines, results):
        out.append(f"{line}\n  -> {c.corrected if isinstance(c, Correction) else 'error: ' + str(c)}")
    return "\n".join(out)


def agent_email(chains, request: str) -> str:
    result = chains["agent"].invoke({"messages": [HumanMessage(request)]}, CONFIG)
    return remove_unrequested_promises(result["messages"][-1].content, request)


def show(func, *args) -> None:
    """Run one task, then print the result in one piece."""
    start = time.time()
    try:
        result = func(*args)
    except Exception as e:
        result = f"Sorry, something went wrong: {e}"
    print(f"\n{BOLD}===== RESULT ====={RESET}\n{result.strip()}")
    print(f"{GREY}({time.time() - start:.0f}s){RESET}")


def pick_model() -> str:
    installed = [m.model for m in ollama.list().models]
    return next((m for m in MODELS if m in installed), MODELS[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", choices=["email", "correct", "agent"])
    p.add_argument("text", nargs="?")
    p.add_argument("-m", "--model", help="Ollama model name")
    p.add_argument("--tone", default="professional")
    p.add_argument("--think", action="store_true", help="show the model's thinking (slower)")
    args = p.parse_args()

    model = args.model or pick_model()
    think = "thinking" in (ollama.show(model).capabilities or []) and args.think
    print(f"[langchain | model: {model} | thinking: {'on' if think else 'off'}]")
    ollama.generate(model=model, prompt="", keep_alive="30m")  # load into RAM now
    chains = build(model, think)

    if args.task and args.text:
        if args.task == "email":
            show(draft_email, chains, args.text, args.tone)
        elif args.task == "correct":
            show(correct_text, chains, args.text)
        else:
            show(agent_email, chains, args.text)
        return

    while True:
        print("\n1) Draft email          2) Revise last email\n"
              "3) Correct English      4) Correct many lines\n"
              "5) Email agent (finds real dates)      q) Quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "1":
            history.clear()                                   # new email = new memory
            request = input("What should the email say?\n> ")
            tone = input("Tone (Enter = professional): ").strip() or "professional"
            show(draft_email, chains, request, tone)
        elif choice == "2":
            change = input("What should change? (e.g. 'make it shorter')\n> ")
            show(draft_email, chains, f"Rewrite the whole email with this change: {change}")
        elif choice == "3":
            show(correct_text, chains, input("Text to correct:\n> "))
        elif choice == "4":
            print("One sentence per line. Empty line = start.")
            lines = []
            while line := input("> ").strip():
                lines.append(line)
            show(correct_many, chains, lines)
        elif choice == "5":
            show(agent_email, chains, input("What should the email say?\n> "))
        else:
            print("Please type 1-5 or q.")


if __name__ == "__main__":
    main()
