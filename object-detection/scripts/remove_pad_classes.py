"""Remove pad_* classes from dataset config.

Since pad classes (155-200) have zero annotations in the dataset,
this only needs to update the YAML config. No label remapping needed.
"""

import yaml
from pathlib import Path


def main():
    config_dir = Path(__file__).parent.parent / "configs"
    src = config_dir / "dataset.yaml"
    dst = config_dir / "dataset_reduced.yaml"

    with open(src) as f:
        config = yaml.safe_load(f)

    # Filter out pad_* classes
    original_names = config["names"]
    filtered = {k: v for k, v in original_names.items() if not v.startswith("pad_")}

    config["names"] = filtered
    config["nc"] = len(filtered)

    with open(dst, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Reduced {len(original_names)} -> {len(filtered)} classes")
    print(f"Removed: {len(original_names) - len(filtered)} pad classes")
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
