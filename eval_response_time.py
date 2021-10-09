import time

import dotenv
import hydra
import numpy
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from src.utils.functions import print_config

dotenv.load_dotenv(override=True)


def main(cfg: DictConfig):
    # Load config
    rank_zero_info(OmegaConf.to_yaml(cfg))

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    model = hydra.utils.instantiate(cfg['model'])

    sec = 60
    input_signal = numpy.zeros((44100 * sec, 2), dtype=np.float)

    start = time.time()
    model.separate_tracks(input_signal,
                          ['vocals', 'drums', 'bass', 'other'],
                          overlap_ratio=cfg['overlap_ratio'],
                          batch_size=cfg['batch_size'])
    end = time.time()

    print('device:{}'.format(model.device))
    print('response time:\n\t{:10.2f}/{}s\n\t{:10.6f}/s'.format(end - start, sec, (end - start) / sec))


@hydra.main(config_path="conf", config_name="model_debug")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == '__main__':
    hydra_entry()
