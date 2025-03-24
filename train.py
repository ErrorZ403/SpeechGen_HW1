from commands_dataset import BinarySpeechCommands, FullSpeechCommands, collate_fn_speech
from model import SpeechCNN

from torchaudio.datasets import SPEECHCOMMANDS

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_results_n_mels(n_mels_list, results, file_name):
    plt.figure(figsize=(12, 4))

    #Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(n_mels_list, [results[n]["test_accuracy"] for n in n_mels_list], 'bo-')
    plt.xlabel('Number of Mel Filterbanks')
    plt.ylabel('Test Accuracy')
    plt.title('n_mels vs Test Accuracy')

    #Flops
    plt.subplot(1, 2, 2)
    plt.plot(n_mels_list, [results[n]["flops"] for n in n_mels_list], 'ro-')
    plt.xlabel('Number of Mel Filterbanks')
    plt.ylabel('FLOPs')
    plt.title('n_mels vs Computational Complexity')

    plt.tight_layout()
    plt.savefig(file_name)

def plot_results_n_groups(n_groups_list, results, file_name='groups_results.png'):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(n_groups_list, 
            [results[n]["avg_epoch_time"] for n in n_groups_list], 'bo-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Groups vs Training Time')

    plt.subplot(1, 3, 2)
    plt.plot(n_groups_list, 
            [results[n]["num_params"] for n in n_groups_list], 'ro-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('Number of Parameters')
    plt.title('Groups vs Model Size')

    plt.subplot(1, 3, 3)
    plt.plot(n_groups_list, 
            [results[n]["test_accuracy"] for n in n_groups_list], 'go-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('Test Accuracy')
    plt.title('Groups vs Test Accuracy')

    plt.tight_layout()
    plt.savefig(file_name)


def train(train_loader, val_loader, test_loader, n_mels, groups, save_path='checkpoints'):
    model = SpeechCNN(n_mels=n_mels, groups=groups)
    trainer = pl.Trainer(
        max_epochs=5,
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir=save_path
    )
    
    num_params = model.get_num_parameters()
    flops = model.get_flops()

    trainer.fit(model, train_loader, val_loader)

    epoch_times = [log["epoch_time"] for log in trainer.logged_metrics.values() 
                if isinstance(log, torch.Tensor) and "epoch_time" in str(log)]

    results = {
        "train_loss": model.trainer.logged_metrics["train_loss_epoch"].item(),
        "val_accuracy": model.trainer.logged_metrics["val_accuracy"].item(),
        "num_params": num_params,
        "flops": flops,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0
    }

    trainer.test(model, test_loader)

    results['test_accuracy'] = model.trainer.logged_metrics["test_accuracy"].item()

    return results


def n_mels_train(train_loader, val_loader, test_loader):
    n_mels_list = [10, 40, 70, 100]

    results = {}

    for n_mels in n_mels_list:
        print(f'Start training with n_mels: {n_mels}')
        results[n_mels] = train(train_loader, val_loader, test_loader, n_mels, 1)

    plot_results_n_mels(n_mels_list, results, 'n_mels_diff_results.png')

    print("Results Summary:")
    for n_mels, metrics in results.items():
        print(f"n_mels = {n_mels}:")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Validation Accuracy: {metrics['val_accuracy']}")
        print(f"Final Train Loss: {metrics['train_loss']}")
        print(f"Average Epoch Time: {metrics['avg_epoch_time']}")
        print(f"Number of Parameters: {metrics['num_params']}")
        print(f"FLOPs: {metrics['flops']}")
        print()

def groups_train(train_loader, val_loader, test_loader):
    n_groups_list = [1, 4, 8, 16]

    results = {}

    for groups in n_groups_list:
        print(f'Start training with n_mels: {groups}')
        results[groups] = train(train_loader, val_loader, test_loader, 40, groups)

    plot_results_n_groups(n_groups_list, results, 'n_groups_diff_results.png')

    print("Results Summary:")
    for groups, metrics in results.items():
        print(f"groups = {groups}:")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Validation Accuracy: {metrics['val_accuracy']}")
        print(f"Final Train Loss: {metrics['train_loss']}")
        print(f"Average Epoch Time: {metrics['avg_epoch_time']}")
        print(f"Number of Parameters: {metrics['num_params']}")
        print(f"FLOPs: {metrics['flops']}")
        print()



if __name__ == '__main__':
    train_dataset = BinarySpeechCommands("training")#FullSpeechCommands(subset='training')
    val_dataset = BinarySpeechCommands("validation")#FullSpeechCommands(subset="validation")
    test_dataset = BinarySpeechCommands("testing")#FullSpeechCommands(subset="testing")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_speech)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)

    n_mels_train(train_loader, val_loader, test_loader)
    groups_train(train_loader, val_loader, test_loader)
