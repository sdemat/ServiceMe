"""Download embedding models into models/. Needs: pip install sentence-transformers

Run: python src/download_models.py [names...] [--force]
"""

import argparse
import sys

import config


def embedding_model_names():
    """Registry models that can be downloaded."""
    return [n for n, spec in config.MODEL_REGISTRY.items() if spec["type"] == "embedding"]


def main():
    parser = argparse.ArgumentParser(description="Download embedding models into models/.")
    parser.add_argument("names", nargs="*", help=f"default: {', '.join(embedding_model_names())}")
    parser.add_argument("--force", action="store_true", help="re-download existing models")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: run  pip install sentence-transformers")
        return 1

    failures = 0
    for name in args.names or embedding_model_names():
        try:
            spec = config.get_model_spec(name)
        except KeyError as e:
            print(f"ERROR: {e}")
            failures += 1
            continue
        if spec["type"] != "embedding":
            print(f"Skipping '{name}': nothing to download.")
            continue

        dest = config.model_dir(name)
        if dest.exists() and any(dest.iterdir()) and not args.force:
            print(f"'{name}' already at {dest} (--force to redo).")
            continue

        print(f"Downloading {spec['source']} ...")
        try:
            model = SentenceTransformer(spec["source"])
            dest.mkdir(parents=True, exist_ok=True)
            model.save(str(dest))
        except Exception as e:
            print(f"ERROR downloading '{name}': {e}")
            failures += 1
            continue

        dim = model.get_sentence_embedding_dimension()
        warn = "" if dim == spec["dim"] else f"  (config says {spec['dim']}; update the registry)"
        print(f"Saved '{name}' to {dest} [{dim}-dim]{warn}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
