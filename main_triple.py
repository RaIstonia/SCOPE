import argparse
import os

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DiscoGR', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--tokenized_dir', type=str, default=None,
                        help='Load pre-tokenized data from this dir, skip tokenizer')
    return parser.parse_known_args()


def _print_run_banner(args, command_line_configs):
    print("=" * 88, flush=True)
    print("[DiscoGR Run]", flush=True)
    print("-" * 88, flush=True)
    print(f"  {'model':>26s} = {args.model}", flush=True)
    print(f"  {'dataset':>26s} = {args.dataset}", flush=True)
    print(f"  {'checkpoint':>26s} = {args.checkpoint}", flush=True)
    print(f"  {'tokenized_dir':>26s} = {args.tokenized_dir}", flush=True)
    print(f"  {'CUDA_VISIBLE_DEVICES':>26s} = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print("-" * 88, flush=True)
    if command_line_configs:
        print("  command-line overrides (priority over config.yaml):", flush=True)
        for k, v in sorted(command_line_configs.items()):
            print(f"    {k:>24s} = {v!r}", flush=True)
    print("=" * 88, flush=True)


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    if args.tokenized_dir is not None:
        command_line_configs['tokenized_dir'] = args.tokenized_dir

    _print_run_banner(args, command_line_configs)

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs,
    )
    pipeline.run()
