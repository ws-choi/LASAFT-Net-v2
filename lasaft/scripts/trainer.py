from pathlib import Path
from typing import List
from warnings import warn

import hydra.utils
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.loggers import WandbLogger

from lasaft.utils.functions import wandb_login, log_hyperparameters
from lasaft.utils.instantiator import HydraInstantiator as HI


def train(cfg: DictConfig):
    if cfg['model']['spec_type'] != 'magnitude':
        cfg['model']['spec2spec']['input_channels'] = 4

    if cfg['trainer']['resume_from_checkpoint'] is None:
        if cfg['seed'] is not None:
            seed_everything(cfg['seed'], workers=True)

    model = hydra.utils.instantiate(cfg['model'])

    if cfg['model']['spec2spec']['last_activation'] != 'identity' and cfg['model']['spec_est_mode'] != 'masking':
        warn('Please check if you really want to use a mapping-based spectrogram estimation method '
             'with a final activation function. ')
    ##########################################################

    # -- checkpoint
    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # -- logger setting
    loggers = None
    if 'logger' in cfg:


        loggers = []

        for logger in cfg['logger']:
            if logger == 'wandb':
                cfg['logger']['wandb']['tags'].append(model.name)
                if 'dev' in model.name:
                    cfg['logger']['wandb']['tags'].append('dev_mode')
                cfg['logger']['wandb']['name'] = '{}_{}_{}'.format(model.name, cfg['seed'], str(model.lr))
                wandb_login(key=cfg['wandb_api_key'])
                logger = hydra.utils.instantiate(cfg['logger']['wandb'])
                logger.watch(model, log='all')
                loggers.append(logger)

    # Trainer
    trainer = HI.trainer(cfg, callbacks=callbacks, logger=loggers, _convert_="partial")
    dp = HI.data_provider(cfg)

    train_dataset, training_dataloader = dp.get_training_dataset_and_loader()
    valid_dataset, validation_dataloader = dp.get_validation_dataset_and_loader()

    if cfg['trainer']['auto_lr_find']:
        lr_find = trainer.tuner.lr_find(model,
                                        training_dataloader,
                                        validation_dataloader,
                                        early_stop_threshold=None,
                                        min_lr=1e-5)

        print(f"Found lr: {lr_find.suggestion()}")
        return None

    if cfg['trainer']['resume_from_checkpoint'] is not None:
        print('resume from the checkpoint')

    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer
    )

    trainer.fit(model, training_dataloader, validation_dataloader)

    return None
