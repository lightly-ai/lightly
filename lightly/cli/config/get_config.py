from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_lightly_config() -> DictConfig:
    config_path = Path(__file__).with_name("config.yaml")
    conf = OmegaConf.load(config_path)
    # TODO(Huan, 05.04.2023): remove this when hydra is completely dropped
    if conf.get("hydra"):
        # This config entry is only for hydra; not referenced in any logic
        del conf["hydra"]
    return conf
