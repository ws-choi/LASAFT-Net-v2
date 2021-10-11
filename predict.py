import time
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf

from evaluator.music_demixing import MusicDemixingPredictor


class LightSAFTPredictor(MusicDemixingPredictor):
    def prediction_setup(self):

        conf_path = Path('./conf/pretrained/v1_light')
        ckpt_path = conf_path.joinpath('epoch=199.ckpt')

        with open(conf_path.joinpath('config.yaml')) as f:
            train_config = OmegaConf.load(f)
            model_config = train_config['model']

            model = hydra.utils.instantiate(model_config).to('cpu')

            try:
                ckpt = torch.load(str(ckpt_path), map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

                print('checkpoint {} is loaded: '.format(ckpt_path))
            except FileNotFoundError:
                print('FileNotFoundError.\n\t {} not exists'.format(ckpt_path))  # issue 10: fault tolerance

        self.model = model

    def separator(self, audio, rate):
        pass

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):
        mix, rate = sf.read(mixture_file_path, dtype='float32')

        batch_size = 64
        start = time.time()
        res = self.model.separate_tracks(input_signal=mix,
                                         targets=['vocals', 'drums', 'bass', 'other'],
                                         overlap_ratio=0.5,
                                         batch_size=batch_size)
        vocals, drums, bass, other = res['vocals'], res['drums'], res['bass'], res['other']
        print('response time: {}'.format(time.time()-start))

        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }

        for target, target_name in zip([vocals, drums, bass, other], ['vocals', 'drums', 'bass', 'other']):
            sf.write(target_file_map[target_name], target, samplerate=44100)


lightsaft_predictor = LightSAFTPredictor()
submission = lightsaft_predictor
submission.run()
print("Successfully completed music demixing...")
