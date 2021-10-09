import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info
from src.source_separation.conditioned.scripts import trainer as trainer
from src.utils.functions import mkdir_if_not_exists, print_config

dotenv.load_dotenv(override=True)


def main(cfg: DictConfig):
    # Load config
    rank_zero_info(OmegaConf.to_yaml(cfg))

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # if cfg['model']['spec_type'] != 'magnitude':
    #     cfg['model']['input_channels'] = 4

    # model = framework(**args)
    model = hydra.utils.instantiate(cfg['model'])
    a = 5


@hydra.main(config_path="conf", config_name="model_debug")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == '__main__':
    hydra_entry()
