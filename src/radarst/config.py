"""Path/configuration validation; fixed algorithm constants are not tuned here."""
from pathlib import Path
import yaml

DEFAULTS = dict(profile="external_first", retention=.97, expression_weight=.8,
                expected_gain_weight=.2, cross_risk_penalty=.025,
                selector_attempts=96, harm_weight=.5, extra_utility_budget=.01,
                python_hash_seed="0")


def read_config(path=None):
    result = DEFAULTS.copy()
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(result)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        result.update(loaded)
    if result["profile"] not in {"internal", "external_first", "external_second"}:
        raise ValueError("Unsupported existing algorithm profile")
    for key, value in DEFAULTS.items():
        if key != "profile" and result[key] != value:
            raise ValueError(f"This implementation requires {key}={value!r}; no algorithm change is applied")
    return result


def new_output(path):
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    return output
