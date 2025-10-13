# src/ui_streamlit_theme.py
from functools import lru_cache
from pathlib import Path
import yaml

def _resolve_yaml_path(yaml_path: str) -> Path:
    p = Path(yaml_path)
    if p.is_absolute() and p.exists():
        return p
    here = Path(__file__).parent
    project = here.parent
    candidates = [
        p,
        Path.cwd() / yaml_path,
        here / yaml_path,
        project / yaml_path,
        Path.cwd() / "configs" / "ui_streamlit_theme.yaml",
        Path.cwd() / "src" / "configs" / "ui_streamlit_theme.yaml",
        here / "configs" / "ui_streamlit_theme.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Không tìm thấy YAML: {yaml_path}")

@lru_cache(maxsize=4)
def get_cfg(yaml_path: str = "configs/ui_streamlit_theme.yaml") -> dict:
    path = _resolve_yaml_path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg
