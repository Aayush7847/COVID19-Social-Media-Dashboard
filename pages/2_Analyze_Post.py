import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ui

ui.render_shell(1)
ui.render_analyzer("an", detailed=True)
