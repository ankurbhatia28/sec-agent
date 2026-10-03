"""`secrag` command-line entry point."""

import argparse
import os
import sys
import urllib.request
from datetime import datetime

from secrag import llm, tracing

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
DEFAULT_SMOKE_ALIASES = ["claude-haiku", "gpt-luna", "qwen-flash", "qwen-local"]


def provider_available(spec: llm.ModelSpec) -> tuple[bool, str]:
    if spec.provider == "ollama":
        base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        try:
            urllib.request.urlopen(base, timeout=1)
            return True, ""
        except OSError:
            return False, f"Ollama not reachable at {base}"
    env = PROVIDER_ENV.get(spec.provider)
    if env and not os.environ.get(env):
        return False, f"{env} not set"
    return True, ""


def smoke(aliases: list[str]) -> int:
    """One tiny call per provider, all nested under a single Langfuse trace."""
    session = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    messages = [
        {"role": "system", "content": "You answer with a single word."},
        {"role": "user", "content": "What form type is a US public company's annual report?"},
    ]
    ok = 0
    print(f"{'alias':<14}{'status':<8}{'in':>6}{'out':>6}{'cost $':>11}  note")
    with tracing.trace("smoke", session_id=session, tags=["smoke"]):
        for alias in aliases:
            spec = llm.get_model(alias)
            available, why = provider_available(spec)
            if not available:
                print(f"{alias:<14}{'skip':<8}{'':>6}{'':>6}{'':>11}  {why}")
                continue
            try:
                r = llm.complete(alias, messages, name=f"smoke:{alias}")
            except Exception as e:  # report and keep going: smoke covers several providers
                print(
                    f"{alias:<14}{'FAIL':<8}{'':>6}{'':>6}{'':>11}  {type(e).__name__}: {e}"[:200]
                )
                continue
            ok += 1
            cost = "n/a" if r.cost_usd is None else f"{r.cost_usd:.6f}"
            note = ("[response cache] " if r.response_cache_hit else "") + r.text.strip()[:40]
            print(
                f"{alias:<14}{'ok':<8}{r.usage['input']:>6}{r.usage['output']:>6}{cost:>11}  {note}"
            )
    tracing.flush()
    print(f"\n{ok}/{len(aliases)} providers succeeded. Langfuse session: {session}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="secrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_smoke = sub.add_parser("smoke", help="one traced call per provider")
    p_smoke.add_argument("aliases", nargs="*", default=DEFAULT_SMOKE_ALIASES)
    args = parser.parse_args()
    if args.cmd == "smoke":
        sys.exit(smoke(args.aliases))
