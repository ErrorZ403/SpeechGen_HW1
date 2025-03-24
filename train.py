from commands_dataset import BinarySpeechCommands, FullSpeechCommands, collate_fn_speech
from model import SpeechCNN

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import plot_results_n_groups, plot_results_n_mels


def train(train_loader, val_loader, test_loader, n_mels, groups, task_type='binary', save_path='checkpoints'):
    model = SpeechCNN(n_mels=n_mels, groups=groups, task_type=task_type)
    trainer = pl.Trainer(
        max_epochs=5,
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir=save_path
    )
    
    num_params = model.get_num_parameters()
    flops = model.get_flops()

    trainer.fit(model, train_loader, val_loader)

    results = {
        "train_loss": model.trainer.logged_metrics["train_loss_epoch"].item(),
        "val_accuracy": model.trainer.logged_metrics["val_accuracy"].item(),
        "num_params": num_params,
        "flops": flops,
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
        print(f"Number of Parameters: {metrics['num_params']}")
        print(f"FLOPs: {metrics['flops']}")
        print()

def groups_train(train_loader, val_loader, test_loader):
    n_groups_list = [1, 2, 4, 8]

    results = {}

    for groups in n_groups_list:
        print(f'Start training with groups: {groups}')
        results[groups] = train(train_loader, val_loader, test_loader, 40, groups)

    plot_results_n_groups(n_groups_list, results, 'n_groups_diff_results.png')

    print("Results Summary:")
    for groups, metrics in results.items():
        print(f"groups = {groups}:")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Validation Accuracy: {metrics['val_accuracy']}")
        print(f"Final Train Loss: {metrics['train_loss']}")
        print(f"Number of Parameters: {metrics['num_params']}")
        print(f"FLOPs: {metrics['flops']}")
        print()

def train_full(train_loader, val_loader, test_loader):
    results = train(train_loader, val_loader, test_loader, 20, 2, task_type='multiclass')
    print(f"Test Accuracy: {results['test_accuracy']}")
    print(f"Validation Accuracy: {results['val_accuracy']}")
    print(f"Final Train Loss: {results['train_loss']}")
    print(f"Number of Parameters: {results['num_params']}")
    print(f"FLOPs: {results['flops']}")
    print()


if __name__ == '__main__':
    train_dataset = FullSpeechCommands(subset='training')#BinarySpeechCommands("training")
    val_dataset = FullSpeechCommands(subset="validation")#BinarySpeechCommands("validation")
    test_dataset = FullSpeechCommands(subset="testing")#BinarySpeechCommands("testing")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_speech)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)

    #n_mels_train(train_loader, val_loader, test_loader)
    #groups_train(train_loader, val_loader, test_loader)
    train_full(train_loader, val_loader, test_loader)
