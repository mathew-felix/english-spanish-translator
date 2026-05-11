import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_SMOKE") != "1",
    reason="Set RUN_MODEL_SMOKE=1 to run the model-backed smoke test.",
)
def test_model_backed_translate_smoke():
    assert Path("best_model.pth").is_file()
    assert Path("data/tokenizer").is_dir()

    from source.inference import translate

    assert translate("The parliamentary session was adjourned.").strip()
