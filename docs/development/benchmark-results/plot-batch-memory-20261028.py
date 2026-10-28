"""Plot measured memory versus throughput gains within each sweep session."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    supported = json.loads((args.folder/'batch-supported-20261028-results.json').read_text())
    assert supported['status'] == 'complete'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, supported['tasks']):
        extended = json.loads((args.folder/f"batch-high-{task['task']}-20261028.json").read_text())
        assert extended['status'] == 'complete'
        for entries, color, marker, label in (
                (task['configurations'], '#007c83', 'o', '65,536 to 1,000,000 batch sweep'),
                (extended['tasks'][0]['configurations'], '#aa6633', 's', 'Extended batch sweep')):
            entries = sorted(entries, key=lambda c: c['batch_size'])
            baseline = next(c['shots_per_second'] for c in entries if c['batch_size'] == 65536)
            x = [max(m['peak_device_used'] for m in c['memory'])/2**30 for c in entries]
            y = [c['shots_per_second']/baseline for c in entries]
            ax.plot(x, y, color=color, marker=marker, label=label, linewidth=1.4, markersize=5)
            if marker == 's':
                ax.annotate(f'{x[-1]:.1f} GiB\n{y[-1]:.2f}x', (x[-1], y[-1]),
                            xytext=(-8, 10), textcoords='offset points', ha='right', fontsize=9, color=color)
        ax.axhline(1, color='#697781', linewidth=.8, linestyle='--')
        ax.set_xscale('log', base=2)
        ax.set_xticks([.5, 1, 2, 4, 8, 16], ['0.5', '1', '2', '4', '8', '16'])
        ax.set_xlim(.32, 21)
        ax.set_ylim(0, 2.25)
        ax.set_title(task['task']+' · '+task['sampling']['method'], loc='left', weight='bold')
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('Maximum per-device memory in separate probe (GiB)')
    for ax in axes[:, 0]:
        ax.set_ylabel('8-GPU throughput gain\nvs. same-sweep batch 65,536')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.suptitle('More allocated GPU memory does not guarantee more throughput',
                 x=.07, ha='left', weight='bold', fontsize=14)
    fig.text(.07, .025, 'Three timed rounds per point; medians normalized to each sweep\'s own baseline. '
             'Shared GHZ server, 8 × RTX 4090.\nMemory includes CUDA context and CuPy reserved buffers; '
             'allocation probes are excluded from timing. Source d128490.', fontsize=9, color='#52606d')
    fig.tight_layout(rect=[0, .09, 1, .9])
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
