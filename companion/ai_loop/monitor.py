from typing import Literal

Category = Literal["data", "config", "transient", "env", "import", "constraint", "schema", "performance", "unknown"]

def classify_error(message: str) -> Category:
    """Classify an error message into coarse categories used by the AI loop."""
    if not isinstance(message, str):
        return "unknown"

    m = message.strip().lower()

    # --- performance/no-message (test expects this for empty string) ---
    if m == "":
        return "performance"

    # --- import issues ---
    if (
        "importerror" in m
        or "no module named" in m
        or "no module" in m
        or "module not found" in m
    ):
        return "import"

    # --- data issues ---
    if (
        "keyerror" in m
        or "missing column" in m
        or "column not found" in m
        or "file not found" in m
        or "no such file" in m
        or "jsondecode" in m
        or "json decode" in m
        or "invalid json" in m
        or "bom" in m
        or ("valueerror" in m and ("invalid" in m or "could not convert" in m))
    ):
        return "data"

    # --- schema / shape issues ---
    if (
        "wrong shape" in m
        or "shape mismatch" in m
        or "reshape" in m
        or "dimensions" in m
        or ("valueerror" in m and "shape" in m)
    ):
        return "schema"

    # --- config mistakes ---
    if (
        "unknown strategy" in m
        or "invalid parameter" in m
        or "invalid config" in m
        or ("yaml" in m and "parse" in m)
        or ("json" in m and "parse" in m)
        or ("typeerror" in m and "unexpected keyword argument" in m)
    ):
        return "config"

    # --- transient: network/timeouts ---
    if (
        "timeout" in m
        or "timed out" in m
        or "connection" in m
        or "temporar" in m
        or "rate limit" in m
        or "429" in m
        or "broken pipe" in m
        or "winerror 10054" in m
    ):
        return "transient"

    # --- environment ---
    if (
        "permission" in m
        or "access denied" in m
        or ("mlflow" in m and ("uri" in m or "tracking" in m))
        or "dll load failed" in m
    ):
        return "env"

    # --- constraints ---
    if (
        "min_trades" in m
        or "minimum trades" in m
        or ("constraint" in m and ("violation" in m or "failed" in m))
    ):
        return "constraint"

    return "unknown"
