import os
import re
import json
import time
import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
PERSONAS_DIR = Path(__file__).parent / "personas"
CACHE_FILE = Path(__file__).parent / "cache_response.json"
CACHE_MAX_AGE_DAYS = 7
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "5"))
RESPONSE_DELAY_SECONDS = int(os.getenv("RESPONSE_DELAY_SECONDS", "5"))

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Persona loading ───────────────────────────────────────────────────────────

def load_personas() -> dict[str, str]:
    personas = {}
    for path in sorted(PERSONAS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        first_line = content.splitlines()[0].lstrip("#").strip()
        name = first_line if first_line else path.stem.replace("_", " ").title()
        personas[name] = content
    return personas


# ── Context cache ─────────────────────────────────────────────────────────────

_CACHE_SYSTEM = """You are an AI assistant managing a warm, realistic family chat room.
The room contains several family members and close relatives, each with their own distinct personality,
life experience, speaking style, and areas of expertise. They are all real, caring people — not robots.

You will be given one of two tasks:

TASK: SCORING — Given a user message and conversation history, rate each persona's relevance (0-10)
for responding. High scores mean the persona has relevant expertise, personality fit, or a natural
stake in the topic. Low scores mean the topic is outside their experience or interest.
Return scores as a JSON array. Only personas scoring at or above the threshold should respond.

TASK: RESPOND AS PERSONA — Stay fully in character as the named persona. Match their tone, vocabulary,
emotional register, and speech patterns exactly as described in their definition. Use the full
conversation history so your reply is coherent with everything said before. Never break character,
never refer to yourself as an AI, and never describe what you are doing — just respond naturally
as that person would in a real family chat.

Persona definitions for all family members are provided below."""


def _personas_block(personas: dict[str, str]) -> str:
    return "\n\n---\n\n".join(f"# {name}\n\n{content}" for name, content in personas.items())


# ── Cache file I/O ────────────────────────────────────────────────────────────

def _read_cache_file() -> dict | None:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None


def _write_cache_file(cache_name: str) -> None:
    CACHE_FILE.write_text(json.dumps({
        "cache_name": cache_name,
        "model": GEMINI_MODEL,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }, indent=2))


def _cache_file_is_fresh(entry: dict) -> bool:
    """True if the entry is < CACHE_MAX_AGE_DAYS old AND was created for the current model."""
    try:
        if entry.get("model") != GEMINI_MODEL:
            return False
        age = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(entry["created_at"])
        return age.days < CACHE_MAX_AGE_DAYS
    except Exception:
        return False


def _verify_cache_live(cache_name: str) -> bool:
    """Ping Gemini to confirm the cache ID still exists."""
    try:
        client.caches.get(name=cache_name)
        return True
    except Exception:
        return False


# ── Cache orchestration ───────────────────────────────────────────────────────

def get_or_refresh_cache(personas: dict[str, str]) -> tuple[str | None, str]:
    """
    Decision tree (no network call unless something fails):
      1. cache_response.json missing              → create new cache
      2. entry older than CACHE_MAX_AGE_DAYS      → create new cache
      3. model in file ≠ current GEMINI_MODEL     → create new cache
      4. Gemini says cache ID is gone             → create new cache
      5. All checks pass                          → reuse existing cache

    Returns (cache_name_or_None, human-readable status string).
    """
    entry = _read_cache_file()

    if entry:
        if not _cache_file_is_fresh(entry):
            reason = (
                f"model changed to `{GEMINI_MODEL}`"
                if entry.get("model") != GEMINI_MODEL
                else f"cache file is older than {CACHE_MAX_AGE_DAYS} days"
            )
            return _create_and_save(personas, reason="refreshed · " + reason)

        if _verify_cache_live(entry["cache_name"]):
            created = entry.get("created_at", "")[:10]
            return entry["cache_name"], f"reused · created {created} · `{entry['cache_name']}`"
        else:
            return _create_and_save(personas, reason="refreshed · previous cache expired in Gemini")

    return _create_and_save(personas, reason="new · no prior cache found")


def _cache_model_name() -> str:
    """The caches API requires the 'models/' prefix; the generate API does not."""
    return GEMINI_MODEL if GEMINI_MODEL.startswith("models/") else f"models/{GEMINI_MODEL}"


def _create_and_save(personas: dict[str, str], reason: str) -> tuple[str | None, str]:
    """Create a brand-new Gemini cache, persist it to disk, return (name, status)."""
    try:
        cache = client.caches.create(
            model=_cache_model_name(),
            config=types.CreateCachedContentConfig(
                system_instruction=_CACHE_SYSTEM,
                contents=[_personas_block(personas)],
                ttl=f"{CACHE_TTL_MINUTES * 60}s",
            ),
        )
        _write_cache_file(cache.name)
        return cache.name, f"{reason} · TTL {CACHE_TTL_MINUTES} min · `{cache.name}`"
    except Exception as exc:
        return None, _cache_error_reason(exc)


def _cache_error_reason(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "resource_exhausted" in msg.lower() or "freetier" in msg.lower().replace("_", ""):
        return "context caching requires a paid Gemini API plan (free tier limit = 0)"
    if "404" in msg or "not found" in msg.lower():
        return f"model `{GEMINI_MODEL}` does not support context caching"
    if "too small" in msg.lower() or "min_total_token_count" in msg.lower():
        import re as _re
        nums = _re.findall(r"\d+", msg)
        detail = f"{nums[0]} tokens, need ≥ {nums[1]}" if len(nums) >= 2 else "content too short"
        return f"persona content too small to cache ({detail}) — add more persona files"
    if "403" in msg or "permission" in msg.lower():
        return "API key does not have permission to use context caching"
    return f"cache creation failed: {msg[:120]}"


# ── Core generate helper ──────────────────────────────────────────────────────

def _generate(
    prompt: str,
    cache_name: str | None = None,
    system_instruction: str = "",
) -> str:
    """
    Single call point for all Gemini requests.
    - With cache_name: attaches cached content (no system_instruction allowed alongside it).
    - Without cache_name: optionally accepts a system_instruction for inline mode.
    """
    config_kwargs: dict = {}
    if cache_name:
        config_kwargs["cached_content"] = cache_name
    elif system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )
    return response.text.strip()


# ── Router agent ──────────────────────────────────────────────────────────────

def score_personas(
    query: str,
    chat_history: list[dict],
    personas: dict[str, str],
    cache_name: str | None,
) -> list[tuple[str, int]]:
    """
    Ask the model to rate each persona's relevance (0-10) for this message.
    Returns a list of (persona_name, score) sorted by score descending,
    filtered to only those at or above SCORE_THRESHOLD.
    Always returns at least one persona (the highest scorer) as a safety net.
    """
    history_text = format_history_for_prompt(chat_history)
    names_list = "\n".join(f"- {n}" for n in personas)

    router_prompt = f"""TASK: SCORE PERSONAS

Rate how relevant each persona is for responding to the user's latest message.
Use a score from 0 (completely irrelevant / wrong expertise) to 10 (perfect fit).

Personas to score:
{names_list}

Conversation so far:
{history_text or "(No previous messages)"}

User's latest message: "{query}"

Reply with ONLY a JSON array — no explanation, no markdown fences:
[
  {{"persona": "Name", "score": 8}},
  {{"persona": "Name", "score": 3}}
]
Include ALL personas."""

    try:
        if cache_name:
            try:
                raw = _generate(router_prompt, cache_name=cache_name)
            except Exception:
                st.session_state.cache_name = None
                st.session_state.cache_status = "expired · switched to inline mode"
                cache_name = None
                raw = _generate(router_prompt)
        else:
            raw = _generate(router_prompt)

        # Pull the JSON array out of the response even if there's surrounding text.
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON array in response")
        data = json.loads(match.group())

        scored = [
            (item["persona"], int(item["score"]))
            for item in data
            if item.get("persona") in personas
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        above = [(n, s) for n, s in scored if s >= SCORE_THRESHOLD]
        # Always keep at least the highest scorer so nobody is left silent.
        return above if above else scored[:1]

    except Exception:
        # Hard fallback: return the first persona with a nominal score.
        return [(next(iter(personas)), 10)]


# ── Response agent ────────────────────────────────────────────────────────────

def generate_response(
    query: str,
    chat_history: list[dict],
    persona_name: str,
    persona_content: str,
    cache_name: str | None,
) -> str:
    history_text = format_history_for_prompt(chat_history)

    if cache_name:
        prompt = f"""TASK: RESPOND AS PERSONA

You are now speaking as {persona_name}. Stay fully in character — match their personality,
tone, and speech style exactly as described in their persona definition above.
Never break character or refer to yourself as an AI.
Use the full conversation history so your answer is coherent with what was already said.

Conversation history:
{history_text or "(This is the first message)"}

User: {query}

Respond as {persona_name}:"""
        try:
            return _generate(prompt, cache_name=cache_name)
        except Exception:
            st.session_state.cache_name = None
            st.session_state.cache_status = "expired · switched to inline mode"
            cache_name = None

    # Inline fallback: persona definition goes into system_instruction.
    system_prompt = f"""You are roleplaying as the following person in a family chat room.
Stay fully in character at all times.

{persona_content}

Rules:
- Respond as {persona_name} would — match their personality, tone, and speech style exactly.
- Use the full conversation history so your answer stays coherent with what was already said.
- Be natural and conversational, not stiff.
- Never break character or refer to yourself as an AI."""

    prompt = f"""Conversation history:
{history_text or "(This is the first message)"}

User: {query}

Respond as {persona_name}:"""

    return _generate(prompt, system_instruction=system_prompt)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_history_for_prompt(chat_history: list[dict]) -> str:
    lines = []
    for msg in chat_history:
        if msg["role"] == "user":
            lines.append(f"User: {msg['content']}")
        else:
            lines.append(f"{msg['persona']}: {msg['content']}")
    return "\n".join(lines)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Family Talk", page_icon="💬", layout="centered")

st.title("💬 Family Talk")
st.caption("A chat room where different family personas respond based on your message.")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    st.error("Set your **GEMINI_API_KEY** in the `.env` file to get started.")
    st.stop()

personas = load_personas()
if not personas:
    st.error(f"No persona `.md` files found in `{PERSONAS_DIR}`. Add some to get started.")
    st.stop()

# ── Cache: check file first, only hit Gemini API if needed ───────────────────
if "cache_name" not in st.session_state:
    with st.spinner("Checking persona cache…"):
        cache_name, cache_status = get_or_refresh_cache(personas)
    st.session_state.cache_name = cache_name
    st.session_state.cache_status = cache_status

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Personas in the room")
    for name, content in personas.items():
        with st.expander(name):
            st.markdown(content)

    st.divider()
    st.subheader("Persona cache")
    if st.session_state.cache_name:
        st.success(st.session_state.cache_status)
    else:
        st.warning(f"Inline mode · {st.session_state.cache_status}")

    st.caption(f"Model: `{GEMINI_MODEL}`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🧑"):
            st.markdown(f"**{msg['persona']}**")
            st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Say something…"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    cache_name = st.session_state.cache_name

    # Score all personas; may return multiple above threshold.
    with st.spinner("Figuring out who wants to reply…"):
        scored = score_personas(
            user_input,
            st.session_state.messages[:-1],
            personas,
            cache_name,
        )

    # Each persona replies in turn, highest score first, with a typing effect.
    for i, (persona_name, score) in enumerate(scored):
        with st.chat_message("assistant", avatar="🧑"):
            name_line = st.markdown(f"**{persona_name}** ✍️ *typing…*")
            body = st.empty()

            reply = generate_response(
                user_input,
                st.session_state.messages[:-1],
                persona_name,
                personas[persona_name],
                cache_name,
            )

            # Replace the typing indicator with the actual message.
            name_line.markdown(f"**{persona_name}**")
            body.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "persona": persona_name}
        )

        # Pause before the next person chimes in (skip after the last reply).
        if i < len(scored) - 1:
            time.sleep(RESPONSE_DELAY_SECONDS)
