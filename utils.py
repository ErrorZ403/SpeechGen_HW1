import matplotlib.pyplot as plt

def plot_results_n_mels(n_mels_list, results, file_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(n_mels_list, [results[n]["test_accuracy"] for n in n_mels_list], 'bo-')
    plt.xlabel('Number of Mel Filterbanks')
    plt.ylabel('Test Accuracy')
    plt.title('n_mels vs Test Accuracy')

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
            [results[n]["num_params"] for n in n_groups_list], 'ro-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('Number of Parameters')
    plt.title('Groups vs Model Size')

    plt.subplot(1, 3, 2)
    plt.plot(n_groups_list, [results[n]["flops"] for n in n_groups_list], 'bo-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('FLOPs')
    plt.title('n_mels vs Computational Complexity')


    plt.subplot(1, 3, 3)
    plt.plot(n_groups_list, 
            [results[n]["test_accuracy"] for n in n_groups_list], 'go-')
    plt.xlabel('Groups Parameter')
    plt.ylabel('Test Accuracy')
    plt.title('Groups vs Test Accuracy')

    plt.tight_layout()
    plt.savefig(file_name)