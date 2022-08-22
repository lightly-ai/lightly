from omegaconf import OmegaConf
import pytest

from lightly.embedding import callbacks


def test_create_summary_callback():
    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=OmegaConf.create(),
    )
    assert summary_cb._max_depth == 99


def test_create_summary_callback__weights_summary():
    # If "weights_summary" is specified, it takes precedence.
    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=OmegaConf.create({"weights_summary": "top"}),
    )
    assert summary_cb._max_depth == 1

    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=OmegaConf.create({"weights_summary": "full"}),
    )
    assert summary_cb._max_depth == -1

    # If "weights_summary" is None or "None", normal config is applied.
    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=OmegaConf.create({"weights_summary": None}),
    )
    assert summary_cb._max_depth == 99

    summary_cb = callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=OmegaConf.create({"weights_summary": "None"}),
    )
    assert summary_cb._max_depth == 99

    with pytest.raises(ValueError):
        callbacks.create_summary_callback(
            summary_callback_config=OmegaConf.create(),
            trainer_config=OmegaConf.create({"weights_summary": "invalid"}),
        )


def test_create_summary_callback__cleans_trainer_config():
    trainer_config = OmegaConf.create({"weights_summary": "None"})
    callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=trainer_config,
    )
    assert "weights_summary" not in trainer_config

    trainer_config = OmegaConf.create({"weights_summary": "top"})
    callbacks.create_summary_callback(
        summary_callback_config=OmegaConf.create({"max_depth": 99}),
        trainer_config=trainer_config,
    )
    assert "weights_summary" not in trainer_config
