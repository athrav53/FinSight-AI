from pathlib import Path
import json

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config folder path
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(filename: str) -> dict:
    """
    Load a JSON configuration file from the config folder.

    Args:
        filename (str): Name of the JSON file.

    Returns:
        dict: Parsed JSON data.
    """

    file_path = CONFIG_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}"
        )

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {filename}"
        ) from e