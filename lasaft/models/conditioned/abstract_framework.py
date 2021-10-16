from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl
import torch

from lasaft.utils.functions import get_optimizer_by_name


class AbstractSeparator(pl.LightningModule, metaclass=ABCMeta):

    def __init__(self, lr, optimizer, initializer):
        super(AbstractSeparator, self).__init__()

        self.lr = lr
        self.optimizer = optimizer
        self.target_names = ['vocals', 'drums', 'bass', 'other']

        if initializer in ['kaiming', 'kaiming_normal']:
            f = torch.nn.init.kaiming_normal_
        elif initializer in ['kaiming_uniform']:
            f = torch.nn.init.kaiming_uniform_
        elif initializer in ['xavier', 'xavier_normal']:
            f = torch.nn.init.xavier_normal_
        elif initializer in ['xavier_uniform']:
            f = torch.nn.init.xavier_uniform
        else:
            raise ModuleNotFoundError

        def init_weights():
            with torch.no_grad():
                for param in self.parameters():
                    if param.dim() > 1:
                        f(param)

        self.initializer = init_weights

    def configure_optimizers(self):
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(self.parameters(), lr=float(self.lr))

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def forward(self, input_signal, input_condition) -> torch.Tensor:
        pass

    @abstractmethod
    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass

    def init_weights(self):
        self.initializer()
