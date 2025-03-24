import torch
from torchaudio.datasets import SPEECHCOMMANDS

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def collate_fn_speech(batch):
    waveforms, labels = zip(*batch)
    max_length = max(w.shape[-1] for w in waveforms)
    target_length = min(max_length, 16000)
    
    padded_waveforms = []
    for w in waveforms:
        if w.shape[-1] < target_length:
            padding = target_length - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, padding))
        elif w.shape[-1] > target_length:
            w = w[..., :target_length]
        padded_waveforms.append(w)
    
    return (
        torch.stack(padded_waveforms),
        torch.tensor(labels)
    )

class BinarySpeechCommands(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(".", download=True, subset=subset)
        self.subset = subset
        self.data = [d for d in self._walker if self._load_label(d) in ["yes", "no"]]
    
    def _load_label(self, file_id):
        return file_id.split("\\")[-2]
    
    def __getitem__(self, index):
        idx = self._walker.index(self.data[index])
        waveform, _, label, _, _  = super().__getitem__(idx)
        label = 1 if label == "yes" else 0
        return (waveform, torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)
    
class FullSpeechCommands(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(".", download=True, subset=subset)
        self.subset = subset
        self.data = [d for d in self._walker if self._load_label(d)]
    
    def _load_label(self, file_id):
        return file_id.split("\\")[-2]
    
    def __getitem__(self, index):
        idx = self._walker.index(self.data[index])
        waveform, _, label, _,  = super().__getitem__(idx)
        print(label)
        label = labels.index(label)
        return (waveform, torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)