from typing import Literal

Category = Literal["data", "config", "transient", "env", "constraint", "unknown"]

def classify_error(message: str) -> Category:
    """Classify an error message into coarse categories used by the AI loop.
    Heuristics are intentionally simple so tests remain stable.
    """
    if not isinstance(message, str):
        return "unknown"

    m = message.lower()

    # --- data issues: missing columns/files, bad JSON/CSV, encoding/BOM ---
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

    # --- config mistakes: bad params, unknown strategy, invalid setting ---
    if (
        "unknown strategy" in m
        or "invalid parameter" in m
        or "invalid config" in m
        or ("yaml" in m and "parse" in m)
        or ("json" in m and "parse" in m)
        or ("typeerror" in m and "unexpected keyword argument" in m)
    ):
        return "config"

    # --- transient: network, timeouts, rate limits, flaky io ---
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

    # --- environment: permissions, paths, mlflow uri, import errors of deps ---
    if (
        "permission" in m
        or "access denied" in m
        or ("mlflow" in m and ("uri" in m or "tracking" in m))
        or "module not found" in m
        or "importerror" in m
        or "dll load failed" in m
        or "no module named" in m
    ):
        return "env"

    # --- constraints: min trades, validation failures specific to tuning ---
    if (
        "min_trades" in m
        or "minimum trades" in m
        or ("constraint" in m and ("violation" in m or "failed" in m))
    ):
        return "constraint"

    return "unknown"
