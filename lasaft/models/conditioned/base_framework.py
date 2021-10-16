import soundfile
import torch
from torch.utils.data import DataLoader

from lasaft.data.musdb_wrapper import SingleTrackSet
from lasaft.models.loss_functions import get_conditional_loss, MultiSourceSpectrogramLoss
from lasaft.models.conditioned.abstract_framework import AbstractSeparator
from lasaft.utils import fourier
from lasaft.utils.fourier import get_trim_length


class AbstractLaSAFTNet(AbstractSeparator):

    def __init__(self,
                 lr, optimizer, initializer,
                 n_fft, num_frame, hop_length, spec_type, spec_est_mode,
                 spec2spec, train_loss, val_loss,
                 name='no_name', norm='bn',
                 query_listen=True,
                 key_listen=True
                 ):
        super(AbstractLaSAFTNet, self).__init__(lr, optimizer, initializer)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim_length = get_trim_length(hop_length)
        self.num_frame = num_frame
        self.name = name
        self.norm = norm
        self.query_listen = query_listen  # Not explicitly used but needed for yaml
        self.key_listen = key_listen  # Not explicitly used but needed for yaml

        assert spec_type in ['magnitude', 'complex']
        assert spec_est_mode in ['masking', 'mapping']
        self.magnitude_based = spec_type == 'magnitude'
        self.masking_based = spec_est_mode == 'masking'
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
        self.stft.freeze()

        self.spec2spec = spec2spec

        self.train_loss = get_conditional_loss(train_loss, n_fft, hop_length)
        self.val_loss = get_conditional_loss(val_loss, n_fft, hop_length)

        self.valid_estimation_dict = {}
        self.init_weights()

    def training_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        loss = self.train_loss(self, mixture_signal, condition, target_signal)

        if isinstance(loss, dict):
            for key in loss.keys():
                self.log(key, loss[key], prog_bar=False, logger=True, on_step=False, on_epoch=True,
                         reduce_fx=torch.mean)

            loss = sum(loss.values())

        self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 reduce_fx=torch.mean)
        return loss

    # Validation Process
    def on_validation_epoch_start(self):
        self.num_val_item = len(self.val_dataloader().dataset)
        for target_name in self.target_names:
            self.valid_estimation_dict[target_name] = {mixture_idx: {}
                                                       for mixture_idx
                                                       in range(14)}

    def validation_step(self, batch, batch_idx):

        mixtures, targets, mixture_ids, window_offsets, input_conditions, target_names = batch

        loss = self.val_loss(self, mixtures, input_conditions, targets)

        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def to_spec(self, input_signal) -> torch.Tensor:
        if self.magnitude_based:
            return self.stft.to_mag(input_signal).transpose(-1, -3)
        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            return spec_complex.transpose(-1, -3)  # *, 2ch, T, N

    def forward(self, input_signal, input_condition) -> torch.Tensor:
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec

        return output_spec

    def separate(self, input_signal, input_condition) -> torch.Tensor:
        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored

    def separate_multisource(self, input_signal, input_conditions, num_targets) -> torch.Tensor:
        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_specs = self.spec2spec.forward_multi_source(input_spec, input_conditions, num_targets)

        if self.masking_based:
            input_spec = input_spec.unsqueeze(-4).repeat(1, num_targets, 1, 1, 1)
            output_specs = input_spec * output_specs
        else:
            pass  # Use the original output_spec

        output_specs = output_specs.transpose(0, 1) # num_target, B, ch, T, F
        restoreds = []

        for output_spec in output_specs:
            output_spec = output_spec.transpose(-1, -3)
            if self.magnitude_based:
                restored = self.stft.restore_mag_phase(output_spec, phase)
            else:
                # output_spec: *, N, T, 2ch
                output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
                restored = self.stft.restore_complex(output_spec)
            restoreds.append(restored)

        return restoreds

    def separate_track(self, input_signal, target, overlap_ratio=0.5, batch_size=1) -> torch.Tensor:

        import numpy as np

        self.eval()

        with torch.no_grad():

            window_length = self.hop_length * (self.num_frame - 1)
            trim_length = get_trim_length(self.hop_length)

            db = SingleTrackSet(input_signal, window_length, trim_length, overlap_ratio)

            assert target in db.source_names
            separated = []

            input_condition = np.array(db.source_names.index(target))
            input_condition = torch.tensor(input_condition, dtype=torch.long, device=self.device).view(1)

            for item, mask in DataLoader(db, batch_size):
                res = self.separate(item.to(self.device), input_condition)
                res = res * mask.to(self.device)
                res = res[:, self.trim_length:-self.trim_length].detach().cpu().numpy()

                separated.append(res)

        separated = np.concatenate(separated)
        if db.is_overlap:
            output = np.zeros_like(input_signal)
            hop_length = db.hop_length
            for i, sep in enumerate(separated):
                to = sep.shape[0]
                if i * hop_length + sep.shape[0] > output.shape[0]:
                    to = sep.shape[0] - (i * hop_length + sep.shape[0] - output.shape[0])
                output[i * hop_length:i * hop_length + to] += sep[:to]
            separated = output

        else:
            separated = np.concatenate(separated, axis=0)

        import soundfile
        soundfile.write('temp.wav', separated, 44100)
        return soundfile.read('temp.wav')[0]

    def separate_tracks(self, input_signal, targets, overlap_ratio=0.5, batch_size=1) -> dict:

        import numpy as np

        if self.spec2spec.is_encoder_conditioned or self.spec2spec.is_encoder_conditioned:
            return {target: self.separate_track(input_signal, target, overlap_ratio, batch_size)
                    for target in targets}

        self.eval()

        with torch.no_grad():

            batch_size //= len(targets)
            window_length = self.hop_length * (self.num_frame - 1)
            trim_length = get_trim_length(self.hop_length)

            db = SingleTrackSet(input_signal, window_length, trim_length, overlap_ratio)

            assert all(target in db.source_names for target in targets)

            separateds = [[] for _ in targets]

            input_conditions = np.array([db.source_names.index(target) for target in targets])
            input_conditions = torch.tensor(input_conditions, dtype=torch.long, device=self.device).view(1, -1)

            for item, mask in DataLoader(db, batch_size):
                results = self.separate_multisource(item.to(self.device),
                                                    input_conditions.repeat(item.shape[0], 1),
                                                    len(targets))

                for res, separated in zip(results, separateds):
                    res = res * mask.to(self.device)
                    res = res[:, self.trim_length:-self.trim_length].detach().cpu().numpy()
                    separated.append(res)

        separateds = [np.concatenate(separated) for separated in separateds]

        new_separateds = []
        if db.is_overlap:
            for separated in separateds:
                output = np.zeros_like(input_signal)
                hop_length = db.hop_length
                for i, sep in enumerate(separated):
                    to = sep.shape[0]
                    if i * hop_length + sep.shape[0] > output.shape[0]:
                        to = sep.shape[0] - (i * hop_length + sep.shape[0] - output.shape[0])
                    output[i * hop_length:i * hop_length + to] += sep[:to]
                new_separateds.append(output)

        else:
            new_separateds = [np.concatenate(separated, axis=0) for separated in separateds]

        res_dict = {}
        for target, separated in zip(targets, new_separateds):
            soundfile.write('{}_temp.wav'.format(target), separated, 44100)
            res_dict[target] = soundfile.read('{}_temp.wav'.format(target))[0]

        return res_dict


class AbstractLaSAFTNetWithMultiSource(AbstractLaSAFTNet):
    def __init__(self,
                 lr, optimizer, initializer,
                 n_fft, num_frame, hop_length, spec_type, spec_est_mode,
                 spec2spec, train_loss, val_loss,
                 name, norm='bn'
                 ):
        super().__init__(lr, optimizer, initializer,
                         n_fft, num_frame, hop_length, spec_type, spec_est_mode,
                         spec2spec, train_loss, val_loss,
                         name, norm)

        assert isinstance(self.train_loss, MultiSourceSpectrogramLoss)

    def forward_multi_source(self, input_signal, conditions, num_targets) -> torch.Tensor:
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec.forward_multi_source(input_spec, conditions, num_targets)

        if self.masking_based:
            output_spec = input_spec.unsqueeze(1) * output_spec

        return output_spec