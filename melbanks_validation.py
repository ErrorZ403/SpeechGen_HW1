import torchaudio
import torch
import matplotlib.pyplot as plt
import torchaudio.transforms
import numpy as np

from melbanks import LogMelFilterBanks

def load_audio(path):
    signal, sr = torchaudio.load(path)

    return signal, sr

def plot_spectrogram(spec, title, save_path):
    if spec.ndim == 3:
        spec = spec.mean(axis = 0)
    plt.figure(figsize=(10, 5))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency Bin")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    signal, sr = load_audio("C:/Users/Nikolay Kalyazin/Desktop/ITMO/Speech/HW1/file_example_WAV_1MG.wav")

    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80,
    )(signal)
    melspec_log = torch.log(melspec + 1e-6)

    logmelbanks = LogMelFilterBanks()(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)

    melspec_np = melspec_log.squeeze().numpy()
    logmelbanks_np = logmelbanks.squeeze().numpy()

    plot_spectrogram(melspec_np, 'TorchAudio MelSpec', 'torchaudio_spec.png')
    plot_spectrogram(logmelbanks_np, 'Custom MelSpec', 'custom_spec.png')
