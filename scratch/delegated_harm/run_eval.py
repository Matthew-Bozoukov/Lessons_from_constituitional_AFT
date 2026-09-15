# ABOUTME: Select experimental recovery or queue workers through the shared eval lifecycle.
# ABOUTME: Run: uv run python -m scratch.delegated_harm.run_eval --name delegated_harm --target <hf> --config <recovery-yaml> --server <ssh>.
import argparse
import sys

from src.eval.run_eval import main as shared_main


def run(target, cfg, out_dir):
    if cfg.get("scaling"):
        from scratch.delegated_harm.scale_worker import run_worker
        return run_worker(target, cfg, out_dir)
    if cfg.get("recovery"):
        from scratch.delegated_harm.recovery import run_recovery
        return run_recovery(target, cfg, out_dir)
    raise ValueError("The experimental entrypoint requires recovery or scaling configuration")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--name")
    parsed, _ = selector.parse_known_args(args)
    if parsed.name is not None and parsed.name != "delegated_harm":
        raise SystemExit("This experimental runner only implements --name delegated_harm")
    shared_main(args, runner=run)


if __name__ == "__main__":
    main()
