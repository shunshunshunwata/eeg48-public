from pathlib import Path
import tempfile

from eeg48.synth import make_synthetic_project, SynthConfig
from eeg48.report import generate_results_card

def test_results_card_generates_html():
    with tempfile.TemporaryDirectory() as td:
        run_dir = make_synthetic_project(Path(td), SynthConfig(n_images=2, n_tables=1, seed=1))
        html_path, index_path = generate_results_card(run_dir)
        assert html_path.exists()
        assert index_path.exists()
        assert "html" in html_path.read_text(encoding="utf-8").lower()
