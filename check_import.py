import os, sys, importlib, traceback

print("CWD:", os.getcwd())
print("Exists src/ui_streamlit_theme.py:", os.path.exists("src/ui_streamlit_theme.py"))

SRC = os.path.join(os.getcwd(), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
print("sys.path[0]:", sys.path[0])

try:
    importlib.import_module("ui_streamlit_theme")   # import KHÔNG có tiền tố src.
    print("OK: imported ui_streamlit_theme")
except Exception as e:
    print("FAILED:", e.__class__.__name__, "→", e)
    traceback.print_exc()
