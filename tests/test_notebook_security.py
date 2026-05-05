import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/colab_training.ipynb")


def test_notebook_has_no_hardcoded_wandb_key():
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    wandb_prefix = "wandb" + "_v1_"
    wandb_assignment = "WANDB_API_KEY = " + '"wandb'

    assert wandb_prefix not in notebook_text
    assert wandb_assignment not in notebook_text


def test_notebook_outputs_are_cleared():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
