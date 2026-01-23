import argparse
import os
import os.path as osp
from pathlib import Path


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _download_hf_dataset(repo_id: str, dest_root: str, revision: str | None = None) -> str:
    """Download a HuggingFace *dataset* repository snapshot to dest_root.

    This is intended to be run on a login/edit node with Internet access.
    The compute node will then load datasets from local `data_dir`.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for offline download. Please install it in your env."
        ) from e

    dest_root = _ensure_dir(dest_root)
    local_dir = osp.join(dest_root, repo_id.replace("/", "__"))
    _ensure_dir(local_dir)

    # snapshot_download will reuse local_dir if already present.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("SCIEVAL_DATA_ROOT", "/data/home/scyb546/datasets"),
        help="Local dataset root directory",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision/commit hash",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HF repo ID",
    )
    args = parser.parse_args()

    assert args.repo_id is not None, "HF repo ID is required"
    local_dir = _download_hf_dataset(args.repo_id, args.root, revision=args.revision)
    print(f"Downloaded {args.repo_id} to: {local_dir}")


if __name__ == "__main__":
    main()
