# src/prompt_loader.py
import yaml
from pathlib import Path

def load_prompts(yaml_path="prompts/prompts.yaml"):
    """Load toàn bộ cấu hình prompt từ file YAML"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def render_system_prompt(cfg, profile_key: str):
    """Trả về system prompt sau khi áp dụng biến và guardrails (nếu có)"""
    profiles = cfg.get("profiles", {})
    overrides = cfg.get("overrides", {})
    variables = overrides.get("variables", {})

    if profile_key not in profiles:
        raise ValueError(f"Profile '{profile_key}' not found in prompts.yaml")

    system_template: str = profiles[profile_key].get("system", "")

    # simple {{var}} replacement
    for k, v in variables.items():
        system_template = system_template.replace(f"{{{{{k}}}}}", v)

    guardrails_file = cfg.get("defaults", {}).get("guardrails_file")
    if guardrails_file and Path(guardrails_file).exists():
        guardrails = Path(guardrails_file).read_text(encoding="utf-8")
        system_template = system_template.replace(
            "Follow the guardrails strictly.",
            f"Follow the guardrails strictly:\n{guardrails}"
        )

    return system_template

def list_profiles(cfg):
    """Trả về dict {profile_key: profile_key} để hiển thị trong selectbox"""
    return {k: k for k in cfg.get("profiles", {}).keys()}
