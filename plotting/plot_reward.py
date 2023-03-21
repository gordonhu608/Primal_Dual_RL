import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_vanilla(data_list, min_len, avg):
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    # plot the rewards
    sns.lineplot(data=data_list, ci='sd')
    # plot the smoothed average rewards
    sns.lineplot(data=avg)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes', fontsize=15)
    plt.ylabel('Reward', fontsize=15)
    plt.title('TD Actor Critic', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    sns.despine()
    plt.tight_layout()
    plt.savefig('Actor_Critic.png', dpi=300)
    plt.show()