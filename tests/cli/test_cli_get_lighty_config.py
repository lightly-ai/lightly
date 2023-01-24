from lightly.cli.config.get_config import get_lightly_config


def test_get_lightly_config() -> None:
    conf = get_lightly_config()
    # Assert some default values
    assert conf.append == False
    assert conf.checkpoint == ""
    assert conf.loader.batch_size == 16
    assert conf.trainer.weights_summary is None
    assert conf.summary_callback.max_depth == 1