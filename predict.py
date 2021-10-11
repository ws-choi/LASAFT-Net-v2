from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from evaluator.music_demixing import MusicDemixingPredictor


lightsaft_predictor = LightSAFTPredictor()
submission = lightsaft_predictor
submission.run()
print("Successfully completed music demixing...")
