
import argparse
import json
import shutil
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
VERSION_FILE = MODELS_DIR / "model_version.json"

ARTIFACTS = [
    "crop_yield_xgb.joblib",
    "crop_yield_model.onnx"
]


def rollback(version):
    version_dir = VERSIONS_DIR / f"v{version}"

    # Validate requested version
    if not version_dir.is_dir():
        raise FileNotFoundError(
            f"Version v{version} does not exist: {version_dir}"
        )

    # Validate all model artifacts before copying anything
    for filename in ARTIFACTS:
        if not (version_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing artifact: {version_dir / filename}"
            )

    # Load version registry
    with open(VERSION_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Restore model artifacts
    for filename in ARTIFACTS:
        source = version_dir / filename
        destination = MODELS_DIR / filename
        shutil.copy2(source, destination)

    # Update current version
    metadata["current_version"] = version
    metadata["status"] = "stable"

    with open(VERSION_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSuccessfully rolled back to v{version}")
    print("Restored artifacts:")
    for filename in ARTIFACTS:
        print(f"- {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Restore a previously saved crop yield model version."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Version to restore, for example 1.0.0"
    )

    args = parser.parse_args()
    rollback(args.version)


if __name__ == "__main__":
    main()
