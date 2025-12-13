from pathlib import Path

# Centralized logs directory for all execution scripts
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
