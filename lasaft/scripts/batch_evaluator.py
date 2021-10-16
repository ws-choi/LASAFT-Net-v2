from pathlib import Path

import hydra
import museval
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from lasaft.utils.functions import wandb_login
from lasaft.utils.instantiator import HydraInstantiator as HI


def batch_eval(cfg: DictConfig):
    eval_dir = Path(cfg['eval_dir'])
    assert eval_dir.exists()

    with open(eval_dir.joinpath('.hydra/config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']
        if 'training' in train_config.keys():
            train_seed = str(train_config['training']['seed'])  # backward compatibility
        else:
            train_seed = str(train_config['seed'])  # backward compatibility

        cfg['model'] = model_config
        model = hydra.utils.instantiate(model_config)

    for checkpoint in eval_dir.joinpath('checkpoints').iterdir():
        if 'ckpt' not in checkpoint.name or 'last' in checkpoint.name:
            continue

        model = model.to('cpu')

        try:
            ckpt = torch.load(str(checkpoint), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

            print('eval: ', checkpoint)
            eval_ckpt(cfg, model, checkpoint, train_seed)
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists'.format(checkpoint))  # issue 10: fault tolerance


def eval_ckpt(cfg: DictConfig, model, ckpt, train_seed):
    # -- logger setting
    if 'logger' in cfg:
        model_name = model.name
        ckpt_name = ckpt.name

        loggers = []

        for logger in cfg['logger']:
            if logger == 'wandb':
                cfg['logger']['wandb']['tags'].append(model_name)
                if 'dev' in model_name:
                    cfg['logger']['wandb']['tags'].append('dev_mode')

                cfg['logger']['wandb']['name'] = '{}_{}_{}'.format(model_name, train_seed, ckpt_name)
                cfg['logger']['wandb']['project'] = 'lasaft_eval'
                cfg['logger']['wandb']['reinit'] = True

                logger = HI.instantiate(cfg['logger']['wandb'])

                wandb_login(key=cfg['wandb_api_key'])

                hparams = {}
                hparams["model"] = cfg['model']
                hparams['overlap_ratio'] = cfg['overlap_ratio']
                # save number of model parameters
                hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
                hparams["model/params_trainable"] = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                hparams["model/params_not_trainable"] = sum(
                    p.numel() for p in model.parameters() if not p.requires_grad
                )

                # send hparams to all loggers
                logger.log_hyperparams(hparams)

                loggers.append(logger)
            else:
                raise NotImplementedError

    else:
        loggers = None

    # DATASET
    dp = HI.data_provider(cfg)

    _, test_data_loader = dp.get_test_dataset_and_loader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg['gpu_id'] is not None:
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(str(cfg['gpu_id']))

    print(device)
    model = model.to(device)

    ##
    overlap_ratio = cfg['overlap_ratio']
    batch_size = cfg['batch_size']

    dataset = test_data_loader.dataset.musdb_reference
    sources = ['vocals', 'drums', 'bass', 'other']

    results = museval.EvalStore(frames_agg='median', tracks_agg='median')

    for idx in range(len(dataset)):

        track = dataset[idx]
        estimation = separate_all(batch_size, model, overlap_ratio, sources, track)

        # Real SDR
        if len(estimation) == len(sources):
            track_length = dataset[idx].samples
            estimated_targets = [estimation[target_name][:track_length] for target_name in sources]
            if track_length > estimated_targets[0].shape[0]:
                raise NotImplementedError
            else:
                estimated_targets_dict = {target_name: estimation[target_name][:track_length] for target_name in
                                          sources}
                track_score = museval.eval_mus_track(
                    dataset[idx],
                    estimated_targets_dict
                )
                score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                    ['target', 'metric'])['score'] \
                    .median().to_dict()
                for logger in loggers:
                    logger.experiment.log(
                        {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()},
                        step=idx)
                    logger.save()
                if len(loggers) < 1:
                    print(track_score)

                results.add_track(track_score)

    for logger in loggers:
        result_dict = results.df.groupby(
            ['track', 'target', 'metric']
        )['score'].median().reset_index().groupby(
            ['target', 'metric']
        )['score'].median().to_dict()
        logger.experiment.log(
            {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
        )
        logger.experiment.log({'test': 1})
        logger.close()

        if isinstance(logger, WandbLogger):
            logger.experiment.finish()

    if len(loggers) < 1:
        print(results)

    return None


def separate_all(batch_size, model, overlap_ratio, sources, track):
    return model.separate_tracks(track.audio, sources, overlap_ratio, batch_size)