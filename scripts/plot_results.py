"""
Plot the task return against the number of exploration steps (mean +- std over exploration runs), as in
Figs. 3 and 4 of the paper.

    python scripts/plot_results.py --env CoppeliaRLBench-v2 --methods LapMCEnt MCDropRenyi MAX PERX Random
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {'LapMCEnt': 'red', 'LapMCRenyi': 'orange', 'MCDropEnt': 'blue', 'MCDropRenyi': 'purple',
          'MAX': 'limegreen', 'EnsEnt': 'green', 'PERX': 'gold', 'Random': 'brown'}


def load_runs(log_dir, env, method):
    files = glob.glob(os.path.join(log_dir, env, method, '*', 'evaluation_*', 'performance_data.csv'))
    return [pd.read_csv(f, index_col=0) for f in sorted(files)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='MagellanHalfCheetah-v2')
    parser.add_argument('--methods', nargs='+', default=list(COLORS))
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--out-dir', default='plots')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {m: load_runs(args.log_dir, args.env, m) for m in args.methods}
    runs = {m: r for m, r in runs.items() if r}
    if not runs:
        raise SystemExit(f'no performance_data.csv found in {args.log_dir}/{args.env}/<method>/<run>/evaluation_*/')

    columns = [c for c in next(iter(runs.values()))[0].columns if c.endswith('_performance')]
    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for method, dfs in runs.items():
            data = pd.concat([df.set_index('step')[column] for df in dfs], axis=1)
            mean, std = data.mean(axis=1), data.std(axis=1).fillna(0)
            color = COLORS.get(method)
            ax.plot(mean.index, mean, color=color, label=f'{method} ({len(dfs)} runs)')
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.1)

        ax.set_title(column.replace('_performance', '').replace('_', ' ').capitalize(), fontsize=16)
        ax.set_xlabel('Exploration steps', fontsize=14)
        ax.set_ylabel('Return', fontsize=14)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        filename = os.path.join(args.out_dir, f'{args.env}_{column}.png')
        fig.savefig(filename)
        plt.close(fig)
        print(f'saved {filename}')


if __name__ == '__main__':
    main()
