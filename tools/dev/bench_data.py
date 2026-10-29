"""Share expensive generated bench data through a private Hugging Face dataset.

    python tools/dev/bench_data.py status | pull [--only ROOT] [--dry-run] | push [--dry-run]

Only generated predictions travel (the names the code reads, below);
ground truth and images keep coming from their pinned upstream URLs. The
committed manifest, bench-data.json, pins the dataset commit and every
artifact's SHA-256, and `pull` verifies both. A directory travels as one
deterministic tar. Needs a token for the private repo (HF_TOKEN or
`hf auth login`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from coco_val_cache import file_sha256
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from real_predictions_cache import (
    cache_root,
    detr_resnet50_cache_filename,
    lvis_detector_cache_filename,
    mask2former_ade_cache_dirname,
    mask2former_panoptic_cache_dirname,
    rfdetr_cache_filename,
    vitpose_cache_filename,
)

MANIFEST = Path(__file__).with_name("bench-data.json")


def jittered_root() -> Path:
    # bench/bench/harness/paths.py::bench_cache_root; bench is not importable here.
    override = os.environ.get("VERNIER_BENCH_CACHE")
    return (Path(override) if override else Path.home() / ".cache" / "vernier-bench") / "jittered"


ROOTS = {"real-models": cache_root, "jittered": jittered_root}


def uploadable(root: str) -> list[Path]:
    """Artifacts of `root` present here that the current code reads."""
    base = ROOTS[root]()
    if root == "real-models":
        names = [
            rfdetr_cache_filename("nano"),
            rfdetr_cache_filename("segnano"),
            detr_resnet50_cache_filename(),
            vitpose_cache_filename(),
            lvis_detector_cache_filename(),
            mask2former_panoptic_cache_dirname(),
            mask2former_ade_cache_dirname(),
        ]
        return [base / n for n in names if (base / n).exists()]
    return sorted(base.glob("*_jittered_seed*_v*.json")) if base.is_dir() else []


@dataclass(frozen=True)
class Artifact:
    root: str
    name: str
    kind: str  # "file" or "dir"
    sha256: str
    size: int

    @property
    def repo_path(self) -> str:
        return f"{self.root}/{self.name}" + (".tar" if self.kind == "dir" else "")


def write_tar(src: Path, dest: Path) -> None:
    """Pack `src` reproducibly: sorted members, zeroed owners and times."""
    with tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(src.rglob("*")):
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(src)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if info.isfile():
                with path.open("rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)


def load_manifest() -> tuple[str, str, dict[tuple[str, str], Artifact]]:
    data = json.loads(MANIFEST.read_text())
    artifacts = {(a["root"], a["name"]): Artifact(**a) for a in data["artifacts"]}
    return data["repo"], data["revision"], artifacts


def save_manifest(repo: str, revision: str, artifacts: dict[tuple[str, str], Artifact]) -> None:
    data = {
        "repo": repo,
        "revision": revision,
        "artifacts": [asdict(a) for _, a in sorted(artifacts.items())],
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def cmd_status(_: argparse.Namespace) -> None:
    repo, revision, artifacts = load_manifest()
    print(f"{repo}@{revision}")
    for (root, name), artifact in sorted(artifacts.items()):
        present = (ROOTS[root]() / name).exists()
        print(f"  {'ok  ' if present else 'MISS'} {root}/{name} ({artifact.size >> 20} MiB)")
    for root in ROOTS:
        for path in uploadable(root):
            if (root, path.name) not in artifacts:
                print(f"  NEW  {root}/{path.name} (not pushed)")


def fetch(repo: str, revision: str, artifact: Artifact, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename stays on one filesystem.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".pull-") as staging:
        fetched = Path(
            hf_hub_download(
                repo,
                artifact.repo_path,
                repo_type="dataset",
                revision=revision,
                local_dir=staging,
            )
        )
        if (got := file_sha256(fetched)) != artifact.sha256:
            raise RuntimeError(f"{artifact.repo_path}: SHA-256 {got}, manifest {artifact.sha256}")
        if artifact.kind == "dir":
            unpacked = Path(staging) / "unpacked"
            with tarfile.open(fetched) as tar:
                if sys.version_info >= (3, 12):
                    tar.extractall(unpacked, filter="data")
                else:
                    tar.extractall(unpacked)  # SHA-256 verified above
            unpacked.rename(dest)
        else:
            fetched.rename(dest)


def cmd_pull(args: argparse.Namespace) -> None:
    repo, revision, artifacts = load_manifest()
    todo = [
        (artifact, ROOTS[root]() / name)
        for (root, name), artifact in sorted(artifacts.items())
        if (args.only in (None, root)) and not (ROOTS[root]() / name).exists()
    ]
    for artifact, dest in todo:
        print(f"pull {artifact.repo_path} ({artifact.size >> 20} MiB) -> {dest}")
    if args.dry_run or not todo:
        return
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(fetch, repo, revision, a, d) for a, d in todo]:
                future.result()
    except HfHubHTTPError as err:
        sys.exit(f"download failed ({err}); is HF_TOKEN set with access to {repo}?")


def cmd_push(args: argparse.Namespace) -> None:
    repo, _, artifacts = load_manifest()
    updated = dict(artifacts)
    with tempfile.TemporaryDirectory(dir=cache_root().parent, prefix=".push-") as staging:
        operations: list[CommitOperationAdd] = []
        for root in ROOTS:
            for path in uploadable(root):
                # Names embed the model revision or generator version, so a
                # pushed name never needs re-uploading.
                if (root, path.name) in artifacts:
                    continue
                if path.is_dir():
                    upload, kind = Path(staging) / f"{path.name}.tar", "dir"
                    write_tar(path, upload)
                else:
                    upload, kind = path, "file"
                size = upload.stat().st_size
                artifact = Artifact(root, path.name, kind, file_sha256(upload), size)
                print(f"push {artifact.repo_path} ({artifact.size >> 20} MiB)")
                operations.append(CommitOperationAdd(artifact.repo_path, str(upload)))
                updated[(root, path.name)] = artifact
        if not operations or args.dry_run:
            print("nothing new to push" if not operations else "dry run: nothing uploaded")
            return
        commit = HfApi().create_commit(
            repo,
            operations=operations,
            commit_message=f"Add {len(operations)} bench data artifacts",
            repo_type="dataset",
        )
    save_manifest(repo, commit.oid, updated)
    print(f"pushed {repo}@{commit.oid}; commit {MANIFEST.name} to share it")


def main() -> None:
    parser = argparse.ArgumentParser(description="Share generated bench data via Hugging Face.")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    pull = sub.add_parser("pull")
    pull.add_argument("--only", choices=sorted(ROOTS))
    pull.add_argument("--dry-run", action="store_true")
    pull.set_defaults(func=cmd_pull)
    push = sub.add_parser("push")
    push.add_argument("--dry-run", action="store_true")
    push.set_defaults(func=cmd_push)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
