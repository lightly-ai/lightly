from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_lightly_config() -> DictConfig:
    config_path = Path(__file__).with_name("config.yaml")
    conf = OmegaConf.load(config_path)
    return conf
