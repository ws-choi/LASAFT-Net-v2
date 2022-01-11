from pathlib import Path
import pkg_resources
import hydra
import torch
from omegaconf import OmegaConf

def get_v2_large_709():

    conf_path = pkg_resources.resource_filename('lasaft', 'pretrained/v2_large/')
    conf_path = Path(conf_path)
    ckpt_path = conf_path.joinpath('epoch=709.ckpt')

    with open(conf_path.joinpath('config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']

        model = hydra.utils.instantiate(model_config).to('cpu')

        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

            print('checkpoint is loaded '.format(ckpt_path))
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

    return model


def get_mdx_light_v2_699():

    conf_path = pkg_resources.resource_filename('lasaft', 'pretrained/v2_light/')
    conf_path = Path(conf_path)
    ckpt_path = conf_path.joinpath('epoch=669.ckpt')

    with open(conf_path.joinpath('config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']

        model = hydra.utils.instantiate(model_config).to('cpu')

        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

            print('checkpoint is loaded '.format(ckpt_path))
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

    return model
