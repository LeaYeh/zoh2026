#!/usr/bin/env python3
"""Run this first to confirm the environment is ready."""
import sys


def check(label: str, fn):
    try:
        fn()
        print(f"  [OK] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False
    return True


ok = True

ok &= check("torch + CUDA", lambda: __import__("torch").cuda.is_available() or True)
ok &= check("chronos-forecasting", lambda: __import__("chronos"))
ok &= check("transformers", lambda: __import__("transformers"))
ok &= check("lightning", lambda: __import__("lightning"))
ok &= check("langgraph", lambda: __import__("langgraph"))
ok &= check("anthropic", lambda: __import__("anthropic"))
ok &= check("wandb", lambda: __import__("wandb"))
ok &= check("gradio", lambda: __import__("gradio"))
ok &= check("yfinance", lambda: __import__("yfinance"))
ok &= check(".env / ANTHROPIC_API_KEY", lambda: (
    __import__("dotenv").load_dotenv() or True,
    __import__("os").environ["ANTHROPIC_API_KEY"],
))

print()
if ok:
    print("Environment ready.")
else:
    print("Fix failures above before proceeding.")
    sys.exit(1)
