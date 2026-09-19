"""Download HF checkpoints into the postax weight cache from a node with
internet access (a cluster login node), so offline compute jobs find them.

    uv run python scripts/prefetch.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
    cluv run tamia python scripts/prefetch.py Qwen/Qwen3-30B-A3B

The destination defaults to $POSTAX_WEIGHTS_DIR, then $SCRATCH/postax/weights
on clusters, then ~/.cache/postax/weights; scripts/job.sh uses the same rule
for model.weights_dir. Note `cluv run` does not apply [tool.cluv.env], so the
$SCRATCH fallback is what makes login-node prefetches and jobs agree.
"""
import argparse
import os
from pathlib import Path

if "HF_HOME" not in os.environ and "SCRATCH" in os.environ:
    # Match HF_HOME in [tool.cluv.env] so tokenizer/dataset caches are found offline.
    os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], "huggingface")

from huggingface_hub import snapshot_download


def default_weights_dir():
    if "POSTAX_WEIGHTS_DIR" in os.environ:
        return os.environ["POSTAX_WEIGHTS_DIR"]
    if "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "postax", "weights")
    return "~/.cache/postax/weights"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_ids", nargs="+", help="HF repo ids, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--weights-dir",
        default=default_weights_dir(),
        help="Weight cache root (default: $POSTAX_WEIGHTS_DIR, $SCRATCH/postax/weights, or ~/.cache/postax/weights)")
    args = parser.parse_args()
    weights_dir = Path(args.weights_dir).expanduser()
    for model_id in args.model_ids:
        local_dir = weights_dir / model_id
        print(f"{model_id} -> {local_dir}")
        snapshot_download(repo_id=model_id, local_dir=local_dir)


if __name__ == "__main__":
    main()
