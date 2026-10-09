"""
Run the exploration and/or evaluation of several methods and seeds.

    # explore with 3 seeds per method
    python scripts/run_experiments.py --env CoppeliaRLBench-v2 --methods LapMCEnt MCDropRenyi MAX --stage explore

    # evaluate every exploration run found in logs/<env>/<method>/
    python scripts/run_experiments.py --env CoppeliaRLBench-v2 --methods LapMCEnt MCDropRenyi MAX --stage evaluate

Extra sacred options can be appended after `--`, e.g. `-- n_exploration_steps=10000`.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ['LapMCEnt', 'LapMCRenyi', 'MCDropEnt', 'MCDropRenyi', 'MAX', 'EnsEnt', 'PERX', 'Random']


def run(options):
    cmd = [sys.executable, 'main.py', 'with'] + options
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='MagellanHalfCheetah-v2',
                        choices=['MagellanHalfCheetah-v2', 'CoppeliaRLBench-v2'])
    parser.add_argument('--methods', nargs='+', default=['LapMCEnt'], choices=METHODS)
    parser.add_argument('--stage', default='all', choices=['explore', 'evaluate', 'all'])
    parser.add_argument('--seeds', type=int, default=3, help='number of exploration runs per method')
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='extra sacred options, after --')
    args = parser.parse_args()
    extra = [e for e in args.extra if e != '--']

    for method in args.methods:
        common = [f'env_name={args.env}', f'method={method}', f'log_dir={args.log_dir}'] + extra

        if args.stage in ('explore', 'all'):
            for _ in range(args.seeds):
                run(['mode=explore'] + common)

        if args.stage in ('evaluate', 'all'):
            method_dir = os.path.join(ROOT, args.log_dir, args.env, method)
            for run_name in sorted(os.listdir(method_dir)):
                run_dir = os.path.join(args.log_dir, args.env, method, run_name)
                if os.path.isdir(os.path.join(ROOT, run_dir)):
                    run(['mode=evaluate', f'run_dir={run_dir}'] + common)


if __name__ == '__main__':
    main()
