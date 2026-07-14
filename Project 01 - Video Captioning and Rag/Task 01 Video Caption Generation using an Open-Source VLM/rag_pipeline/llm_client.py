"""
LLM wrapper for the RAG answer-synthesis step.

Supports two providers, switched via config.LLM_PROVIDER (env var LLM_PROVIDER):
  - "groq"      (default) - FREE, no credit card, fast. Get a key at console.groq.com
  - "anthropic" - paid, higher quality (Claude). Get a key at console.anthropic.com

generate_answer() below gives one unified interface to the rest of the pipeline
regardless of which provider is active.
"""

import config

SYSTEM_PROMPT = """\
You are a field-report assistant for an inspection video pipeline. You answer \
questions about a set of inspected videos using ONLY the context provided below \
(a structured index of every video plus detailed excerpts for the most relevant ones). \
Do not use outside knowledge about the world beyond what's in the context.

Rules:
- If the question asks to "show", "find", or "list" videos (or is otherwise about \
  which videos match something), your answer MUST list each matching video as:
    - <exact video filename> - <its summary caption> - <one short line on why it matches>
  Do not just describe the criteria in prose; name the actual matching videos.
- Judge whether a video matches descriptive details (e.g. "wearing a helmet", \
  "carrying a battery", "performing an activity") by reading its summary caption, \
  detailed analysis, and frame captions in the context - these details won't always \
  appear as a formal "Expected Label", so rely on the actual described content.
- If the answer requires counting, listing, or comparing across videos, use the \
  full structured index (it covers every video, not just the detailed excerpts).
- If asked about model/pipeline accuracy, precision, recall, or false positives/negatives, \
  use the overall metrics block if present.
- If nothing in the context genuinely matches, say so plainly - don't force a match or guess.
- Keep prose tight; prefer bullet points whenever listing more than one video.
"""

_groq_client = None
_anthropic_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and add it to your .env file."
            )
        from groq import Groq

        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file, "
                "or set LLM_PROVIDER=groq to use the free option instead."
            )
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _generate_groq(query: str, context: str) -> str:
    client = _get_groq_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
        ],
    )
    return completion.choices[0].message.content


def _generate_anthropic(query: str, context: str) -> str:
    client = _get_anthropic_client()
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def generate_answer(query: str, context: str) -> str:
    if config.LLM_PROVIDER == "anthropic":
        return _generate_anthropic(query, context)
    return _generate_groq(query, context)
