"""Generate paper figures from experiment results."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FIGDIR = Path(__file__).parent


def fig2_alfworld_per_task():
    """Bar chart: ALFWorld per-task type comparison."""
    tasks = ['clean', 'cool', 'examine', 'heat', 'put', 'puttwo']
    labels = ['Clean', 'Cool', 'Examine', 'Heat', 'Put', 'PutTwo']

    baseline = [80.7, 100, 100, 82.6, 100, 88.2]
    reflexion = [83.9, 100, 100, 91.3, 100, 94.1]
    reactfm = [93.5, 100, 94.4, 91.3, 100, 94.1]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars1 = ax.bar(x - width, baseline, width, label='ReAct Baseline', color='#4472C4', edgecolor='white')
    bars2 = ax.bar(x, reflexion, width, label='Reflexion', color='#ED7D31', edgecolor='white')
    bars3 = ax.bar(x + width, reactfm, width, label='React_FM (E2)', color='#70AD47', edgecolor='white')

    ax.set_ylabel('Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 105)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(FIGDIR / 'alfworld_per_task.pdf')
    fig.savefig(FIGDIR / 'alfworld_per_task.png')
    plt.close()
    print('Generated: alfworld_per_task.pdf')


def fig3_learning_curve():
    """Line chart: Learning curve across epochs for the multi-epoch benchmarks."""
    # ALFWorld
    epochs_alf = [0, 1, 2, 3]
    baseline_alf = [91.0, 91.0, 91.0, 91.0]
    reflexion_alf = [89.6, 94.0, 94.0, 94.0]  # T0, T1
    reactfm_alf = [91.0, 91.8, 95.5, 96.3]  # baseline, E1, E2, E3

    # WebShop
    epochs_ws = [0, 1, 2]
    baseline_ws = [38.0, 38.0, 38.0]
    reactfm_ws = [38.0, 40.0, 44.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # ALFWorld
    ax1.plot(epochs_alf, baseline_alf, 'o--', color='#4472C4', label='Baseline', markersize=5)
    ax1.plot([0, 1], [89.6, 94.0], 's--', color='#ED7D31', label='Reflexion', markersize=5)
    ax1.plot(epochs_alf, reactfm_alf, 'D-', color='#70AD47', label='React_FM', markersize=5, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('ALFWorld', fontsize=11)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_ylim(88, 98)
    ax1.legend(fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3)

    # WebShop
    ax2.plot(epochs_ws, baseline_ws, 'o--', color='#4472C4', label='Baseline', markersize=5)
    ax2.plot(epochs_ws, reactfm_ws, 'D-', color='#70AD47', label='React_FM', markersize=5, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('WebShop', fontsize=11)
    ax2.set_xticks([0, 1, 2])
    ax2.set_ylim(30, 50)
    ax2.legend(fontsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'learning_curve.pdf')
    fig.savefig(FIGDIR / 'learning_curve.png')
    plt.close()
    print('Generated: learning_curve.pdf')


if __name__ == '__main__':
    fig2_alfworld_per_task()
    fig3_learning_curve()
    print('All figures generated.')
