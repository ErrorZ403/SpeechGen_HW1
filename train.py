from commands_dataset import BinarySpeechCommands, collate_fn_speech
from model import SpeechCNN

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(train_loader, val_loader, test_loader, n_mels, save_path='checkpoints'):
    model = SpeechCNN(n_mels=n_mels)
    trainer = pl.Trainer(
        max_epochs=2,
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

def plot_results(n_mels_list, results, file_name):
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

if __name__ == '__main__':
    train_dataset = BinarySpeechCommands("training")
    val_dataset = BinarySpeechCommands("validation")
    test_dataset = BinarySpeechCommands("testing")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_speech)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_speech)

    n_mels_list = [20, 40, 60, 80]

    results = {}

    for n_mels in n_mels_list:
        print(f'Start training with n_mels: {n_mels}')
        results[n_mels] = train(train_loader, val_loader, test_loader, n_mels)

    plot_results(n_mels_list, results, 'n_mels_diff_results.png')

    print("Results Summary:")
    for n_mels, metrics in results.items():
        print(f"n_mels = {n_mels}:")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Validation Accuracy: {metrics['val_accuracy']}")
        print(f"Final Train Loss: {metrics['train_loss']}")
        print(f"Number of Parameters: {metrics['num_params']}")
        print(f"FLOPs: {metrics['flops']}")
        print()