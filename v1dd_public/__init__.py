from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
ARTIFACT_DIR = REPO_ROOT / 'artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)