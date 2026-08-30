"""Remove the leftovers of optimization runs that did not finish.

A completed run writes one artifact into each of four folders — see
`RunPaths` in lbm/src/utils/paths.py, which is the single source of that
naming:

    results/animations/<stem>.mp4
    results/arrays/<stem>.npz
    results/logs/<stem>.txt        (+ an optional <stem>_fields.txt)
    results/plots/<stem>.png

Killing a run part-way leaves some of those behind and not others, and the
survivors are worthless on their own: an animation with no log is a movie
of parameters you can no longer identify, and a log with no arrays is a
description of a result you no longer have. So the rule here is simply
that a run is junk unless all four are present.

That rule is also the interface. To throw a run away, delete any ONE of
its files — whichever folder you happen to be looking at — then run this,
and it finds the other three and removes them too.

    python -m topopt.src.utils.cleanup                # list, then confirm
    python -m topopt.src.utils.cleanup --dry-run      # list only
    python -m topopt.src.utils.cleanup --yes          # no prompt
    python -m topopt.src.utils.cleanup --root other/results

Subdirectories are ignored, never descended into: only files sitting
directly in the four folders are treated as run artifacts.
"""

import argparse
from pathlib import Path

# Mirrors RunPaths._candidates. A run is complete when it has a primary
# artifact in every one of these.
ARTIFACT_DIRS = ("animations", "arrays", "logs", "plots")

# <stem>_fields.txt rides along with <stem>.txt. It is owned by the stem,
# so it is deleted with it, but it does NOT by itself satisfy the logs
# requirement: a field dump whose run log is gone is exactly the kind of
# orphan this tool exists to remove.
EXTRA_SUFFIXES = ("_fields",)


class Run:
    """One stem, and every file across the four folders that belongs to it."""

    def __init__(self, stem):
        self.stem = stem
        self.files = []                  # [(folder, Path)]
        self.primary = set()             # folders holding a primary artifact

    @property
    def missing(self):
        return [d for d in ARTIFACT_DIRS if d not in self.primary]

    @property
    def complete(self):
        return not self.missing

    @property
    def nbytes(self):
        total = 0
        for _, path in self.files:
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return total


def _classify(filename):
    """(stem, is_primary) for a filename. '<x>_fields.txt' -> ('<x>', False)."""
    stem = Path(filename).stem
    for suffix in EXTRA_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], False
    return stem, True


def index_runs(root="results"):
    """Group every artifact under `root` by the run stem that owns it."""
    root = Path(root)
    runs = {}

    for folder in ARTIFACT_DIRS:
        directory = root / folder
        if not directory.is_dir():
            continue
        # iterdir, not rglob: subdirectories are somebody else's files.
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            stem, is_primary = _classify(path.name)
            run = runs.setdefault(stem, Run(stem))
            run.files.append((folder, path))
            if is_primary:
                run.primary.add(folder)

    return runs


def _subdirectories(root):
    found = []
    for folder in ARTIFACT_DIRS:
        directory = root / folder
        if directory.is_dir():
            found += [p for p in directory.iterdir() if p.is_dir()]
    return found


def _human(nbytes):
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def clean_incomplete_runs(root="results", dry_run=False, assume_yes=False,
                          verbose=True):
    """Delete every artifact of every run missing one of its four files.

    Prints what it intends to remove before removing anything. Returns the
    list of Run objects deleted (or that would be, under dry_run), so this
    is usable as a library call as well as a command.
    """
    root = Path(root)
    if not root.is_dir():
        print(f"No such results directory: {root.resolve()}")
        return []

    runs = index_runs(root)
    incomplete = sorted((r for r in runs.values() if not r.complete),
                        key=lambda r: r.stem)
    kept = sum(1 for r in runs.values() if r.complete)

    print(f"Scanning {root.resolve()}")
    skipped = _subdirectories(root)
    if skipped:
        plural = "y" if len(skipped) == 1 else "ies"
        print(f"  ignoring {len(skipped)} subdirector{plural}")
    print()

    if not incomplete:
        print(f"Nothing to clean. All {kept} runs are complete.")
        return []

    n_files = sum(len(r.files) for r in incomplete)
    n_bytes = sum(r.nbytes for r in incomplete)

    print(f"Incomplete runs to remove: {len(incomplete)} "
          f"({n_files} files, {_human(n_bytes)})\n")
    width = max(len(r.stem) for r in incomplete)
    for run in incomplete:
        print(f"  {run.stem:<{width}}   missing: {', '.join(run.missing)}")
        if verbose:
            for folder, path in run.files:
                print(f"      {folder}/{path.name}")
    print(f"\nComplete runs kept: {kept}")

    if dry_run:
        # Printed output stays ASCII: Windows consoles default to cp1252
        # and mangle anything else.
        print("\nDry run - nothing deleted.")
        return incomplete

    if not assume_yes:
        # isatty() is not reliable here — under some Windows shells it
        # reports a terminal even when stdin is redirected — so the EOF is
        # caught rather than predicted. Either way the answer is no: an
        # unattended run must never delete without --yes.
        try:
            answer = input(f"\nDelete these {n_files} files? [y/N] ")
        except EOFError:
            print("\nNo input available; re-run with --yes to delete. "
                  "Nothing deleted.")
            return []
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. Nothing deleted.")
            return []

    removed = 0
    for run in incomplete:
        for _, path in run.files:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                print(f"  could not delete {path}: {exc}")

    print(f"\nDeleted {removed} files from {len(incomplete)} runs "
          f"({_human(n_bytes)} freed).")
    return incomplete


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m topopt.src.utils.cleanup",
        description="Delete artifacts of runs missing any of their four files.")
    parser.add_argument("--root", default="results",
                        help="results directory (default: results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would go, delete nothing")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--quiet", action="store_true",
                        help="list run names only, not every file")
    args = parser.parse_args(argv)

    clean_incomplete_runs(root=args.root, dry_run=args.dry_run,
                          assume_yes=args.yes, verbose=not args.quiet)


if __name__ == "__main__":
    main()
