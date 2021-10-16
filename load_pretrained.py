from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def get_mdx_light_v2_699():
    conf_path = Path('./conf/pretrained/v2_light')
    ckpt_path = conf_path.joinpath('epoch=669.ckpt')

    with open(conf_path.joinpath('config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']

        model = hydra.utils.instantiate(model_config).to('cpu')

        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

            print('checkpoint {} is loaded: '.format(ckpt_path))
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance
