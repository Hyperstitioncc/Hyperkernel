#!/usr/bin/env python3
"""Known-answer checks against a serving `dsv41 serve --preset eval`: short prompts, then long ones whose answer sits early or mid
prompt, so context lost across prefill chunks shows up as a wrong answer. Standard library only.

    known_answers.py PORT [MODEL]   prints PASS or FAIL per prompt; exit 0 only if all pass
"""
import json
import sys
import urllib.request

PORT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4.1-flash"
FILL = " ".join(f"Line {i}: the river runs past the old mill and the workers count the sacks of grain." for i in range(3000))
SHORT = 60
LONG = 300
PROMPTS = [
    ("capital-short", "What is the capital of France? Answer in one word.", ["Paris"], SHORT),
    ("capital-germany", "What is the capital of Germany? Answer in one word.", ["Berlin"], SHORT),
    ("count-short", "Continue the sequence with the next three numbers only: 78, 79, 80,", ["81", "82", "83"], SHORT),
    ("capital-long", FILL[:30000] + "\n\nIgnore the text above. What is the capital of France? Answer in one word.", ["Paris"], LONG),
    ("passkey-start", "The secret passkey is 48213. Remember it.\n\n" + FILL[:60000] + "\n\nWhat is the secret passkey given at the very start? Answer with the number only.", ["48213"], LONG),
    ("passkey-middle", FILL[:30000] + "\nThe secret passkey is 90751. Remember it.\n" + FILL[30000:90000] + "\n\nWhat is the secret passkey given in the text? Answer with the number only.", ["90751"], LONG),
    ("count-long", FILL[:45000] + "\n\nContinue the sequence with the next three numbers only: 78, 79, 80,", ["81", "82", "83"], LONG),
]


def ask(text, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as reply:
        data = json.load(reply)
    message = data["choices"][0]["message"]
    return (message.get("content") or "") + " " + (message.get("reasoning_content") or ""), data.get("usage", {}).get("prompt_tokens")


passed = 0
for name, text, want, max_tokens in PROMPTS:
    try:
        answer, tokens = ask(text, max_tokens)
        good = all(w in answer for w in want)
        passed += good
        print(f"{name}: prompt_tokens={tokens} {'PASS' if good else 'FAIL'} {answer[:120]!r}", flush=True)
    except Exception as e:
        print(f"{name}: ERROR {e}", flush=True)
print(f"known answers: {passed}/{len(PROMPTS)}", flush=True)
sys.exit(0 if passed == len(PROMPTS) else 1)
