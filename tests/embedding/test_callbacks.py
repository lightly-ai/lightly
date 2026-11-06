from omegaconf import OmegaConf

from lightly.embedding import callbacks


def test_create_summary_callback():
    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
    )
    assert summary_cb._max_depth == 99
