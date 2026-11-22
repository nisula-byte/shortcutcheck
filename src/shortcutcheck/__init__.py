"""ShortcutCheck: label-free metamorphic testing for shortcut learning in malware detectors."""

__version__ = "1.0.0"


def dashboard_app_path() -> str:
    """Path of the Streamlit entry script (for ``streamlit run``)."""
    from pathlib import Path

    return str(Path(__file__).with_name("dashboard_app.py"))
