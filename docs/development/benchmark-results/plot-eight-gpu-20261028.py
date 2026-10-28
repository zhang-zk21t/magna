"""Plot the interleaved 1/4/8-GPU measurements without extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    if data['status'] != 'complete':
        raise ValueError('only completed measurements can be plotted')
    tasks = data['tasks']
    x = np.arange(len(tasks))
    labels = [t['task']+'\n'+t['sampling']['method'] for t in tasks]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ['#52677d', '#4e9ba1', '#00636b']
    for k, mode in enumerate(('single', 'four', 'eight')):
        rates = np.array([t[mode]['shots_per_second']/1e6 for t in tasks])
        low = np.array([t['shots']/t[mode]['max_seconds']/1e6 for t in tasks])
        high = np.array([t['shots']/t[mode]['min_seconds']/1e6 for t in tasks])
        left.bar(x+(k-1)*.25, rates, .25, color=colors[k], label=['1 GPU', '4 GPUs', '8 GPUs'][k])
        left.errorbar(x+(k-1)*.25, rates, yerr=[rates-low, high-rates], fmt='none',
                      ecolor='#263442', capsize=2, linewidth=1)
    left.set_yscale('log')
    left.set_ylabel('Million draws / second (log scale)')
    left.set_title('Measured sampling throughput', loc='left', weight='bold')
    left.set_xticks(x, labels)
    left.legend(frameon=False)
    for k, mode in enumerate(('four', 'eight')):
        values = [t[mode+'_over_single'] for t in tasks]
        at = x+(k-.5)*.34
        right.bar(at, values, .34, color=colors[k+1], label=['4 GPUs', '8 GPUs'][k])
        for pos, value in zip(at, values):
            right.text(pos, value+.08, f'{value:.2f}x', ha='center', fontsize=9, weight='bold')
    right.axhline(8, color='#9c6b35', linestyle='--', linewidth=1, label='Ideal 8x')
    right.axhline(1, color='#7d8790', linewidth=.8)
    right.set_ylim(0, max(9, max(t['eight_over_single'] for t in tasks)+1))
    right.set_ylabel('Speedup over direct single GPU')
    right.set_xticks(x, labels)
    right.set_title('Scaling at equal total draws', loc='left', weight='bold')
    right.legend(frameon=False, ncol=3, fontsize=9)
    for ax in (left, right):
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('magna on GHZ · RTX 4090 · 1 / 4 / 8 GPUs', x=.06, ha='left', weight='bold', fontsize=15)
    fig.text(.06, .025, f"Median of {data['repeats']} interleaved rounds; error bars show observed min/max. "
             f"Batch size {data['batch_size']:,} per GPU.\nWarm runs exclude process startup and first-use JIT. "
             f"Shared server; measured source {data['source_commit'][:7]}.", fontsize=9, color='#52606d')
    fig.tight_layout(rect=[0, .13, 1, .93])
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
