from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        # ...
        # <YOUR CODE GOES HERE>
        
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        # ...
        # <YOUR CODE GOES HERE>
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else float(samplerate // 2)
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        # To access attributes, use self.<parameter_name>
        return F.melscale_fbanks(
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            # make a full and correct function call
            # <YOUR CODE GOES HERE>
            n_freqs = self.n_fft // 2 + 1 if self.onesided else self.n_fft,
            f_min = self.f_min_hz,
            f_max = self.f_max_hz,
            n_mels = self.n_mels,
            sample_rate = self.samplerate,
            norm = self.norm_mel,
            mel_scale = self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(
            # make a full and correct function call
            # <YOUR CODE GOES HERE>
            input = x,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.window_length,
            window = self.window,
            center = self.center,
            pad_mode = self.pad_mode,
            normalized = self.normalize_stft,
            onesided = self.onesided,
            return_complex = self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        # <YOUR CODE GOES HERE>
        # Return log mel filterbanks matrix
        x = self.spectrogram(x)
        x = x.abs().pow(self.power)
        x = torch.matmul(x.transpose(-1, -2), self.mel_fbanks).transpose(-1, -2)

        x = torch.log(x + 1e-6)

        return x
    

if __name__ == "__main__":
    import torchaudio
    torchaudio.set_audio_backend("sox_io")
    signal, sr = torchaudio.load("C:/Users/Nikolay Kalyazin/Desktop/ITMO/Speech/HW1/file_example_WAV_1MG.wav")
    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80,
    )(signal)
    logmelbanks = LogMelFilterBanks()(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
