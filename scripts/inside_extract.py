"""Hard filters for atom text. Deterministic; no model on write."""


def is_tool_dump(text: str) -> bool:
    """Tool stdout, listings, and fetched bodies are attach, not atoms."""
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    if "```" in lower and ("stdout" in lower or "stderr" in lower):
        return True
    if lower.startswith("<!doctype") or lower.startswith("<html"):
        return True
    lines = t.splitlines()
    if len(lines) >= 8 and sum(
        1
        for line in lines
        if line.startswith("-") or line.startswith("drwx") or line.startswith("total ")
    ) >= 6:
        return True
    return False
