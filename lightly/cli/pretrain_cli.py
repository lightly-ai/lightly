import hydra
from omegaconf import DictConfig, OmegaConf
from lightly.cli._helpers import fix_hydra_arguments
from pathlib import Path
from lightly.pretrain import pretrain


@hydra.main(
    **fix_hydra_arguments(
        config_path=str(Path(__file__).parent / ".." / "pretrain" / "configs"),
        config_name="default",
    )
)
def pretrain_cli(cfg: DictConfig) -> None:
    print("=" * 20)
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 20)
    pretrain.pretrain_from_cfg(cfg)


def entry():
    pretrain_cli()
