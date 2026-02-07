import subprocess

def test_cli_help():
    p = subprocess.run(["eeg48", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "EEG48" in p.stdout

def test_cli_run_all_help():
    p = subprocess.run(["eeg48", "run-all", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "master report" in p.stdout.lower()
