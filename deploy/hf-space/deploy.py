"""Upload the files the Docker Space needs to <username>/evidencerag.

Log in first (`hf auth login`). Use --dry-run to list the files and upload nothing.
"""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

SPACE_NAME = "evidencerag"
CORPUS = Path("data/evaluation/v2/corpus")
SKIPPED_DIRS = {".venv", "__pycache__", "tests", "node_modules", "dist", ".pytest_cache", ".vite"}
SKIPPED_FILES = {".env"}
# Last line of defence: nothing under these ever goes to a public Space.
NEVER = ("paper", "data/uploads", "data/index", ".superpowers", ".env", "reports", "data/evaluation/results")


def walk(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and not SKIPPED_DIRS.intersection(path.relative_to(folder).parts)
        and path.name not in SKIPPED_FILES
        and path.suffix != ".pyc"
    ]


def collect() -> dict[str, Path]:
    """Map each path in the Space repo to its source file."""
    files = {name: HERE / name for name in ("Dockerfile", "start.sh", "README.md")}
    files["LICENSE"] = ROOT / "LICENSE"
    for folder in (Path("backend"), Path("frontend")):
        for path in walk(ROOT / folder):
            files[path.relative_to(ROOT).as_posix()] = path
    for path in sorted((ROOT / CORPUS).iterdir()):
        if path.is_file():
            files[(CORPUS / path.name).as_posix()] = path
    for name in files:
        if name == "paper" or name.startswith(tuple(f"{p}/" for p in NEVER)) or name in NEVER:
            raise SystemExit(f"Refusing to upload {name}.")
    return files


def stage(files: dict[str, Path], destination: Path) -> None:
    for name, source in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", help="Hugging Face user or organisation; default is the logged-in user")
    parser.add_argument("--dry-run", action="store_true", help="list the files and upload nothing")
    parser.add_argument("--stage", type=Path, help="copy the files into this new, empty folder and upload nothing")
    args = parser.parse_args()

    files = collect()
    if args.stage:
        if args.stage.exists() and any(args.stage.iterdir()):
            raise SystemExit(f"{args.stage} is not empty.")
        stage(files, args.stage)
        print(f"Staged {len(files)} files in {args.stage}.")
        return

    owner = args.username
    if not owner and not args.dry_run:
        from huggingface_hub import HfApi

        owner = HfApi().whoami()["name"]
    repo_id = f"{owner or '<username>'}/{SPACE_NAME}"

    print(f"Files for {repo_id} ({len(files)}):")
    for name in files:
        print(f"  {name}")
    if args.dry_run:
        print("Dry run: nothing was uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage(files, Path(folder))
        api.upload_folder(
            folder_path=folder, repo_id=repo_id, repo_type="space", commit_message="Update the demo"
        )
    print(f"Uploaded {len(files)} files.")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    sys.exit(main())
