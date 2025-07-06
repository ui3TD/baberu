from baberu.tools import file_utils
from baberu.__main__ import APP_NAME

import platformdirs
import yaml

import importlib.resources
import logging
import sys
from pathlib import Path
from typing import Any


def load_config(arg: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file with a fallback mechanism.

    The lookup order is:
    1. Path specified by the commandline argument (not implemented).
    2. `config.yaml` in the project root (for development).
    3. `config.yaml` in the user's config directory (e.g., ~/.config/myapp/).
    4. The default `default_config.yaml` packaged with the application.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    config_logger = logging.getLogger(__name__)

    # 1. Check arg
    if arg and arg.exists():
        with open(arg, 'r') as f:
            return yaml.safe_load(f)

    # 2. Check project root (for local development)
    try:
        project_root = file_utils.get_project_dir()
        dev_config_path = project_root / "config.yaml"
        if dev_config_path.exists():
            config_logger.info(f"Loading config from dev directory: {dev_config_path}")
            with open(dev_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    except FileNotFoundError:
        pass

    # 3. Check user-specific config directory (requires `platformdirs` package)
    user_config_path = Path(platformdirs.user_config_dir(APP_NAME)) / "config.yaml"
    if user_config_path.exists():
        config_logger.info(f"Loading config from user directory: {user_config_path}")
        with open(user_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # 4. Fallback to the packaged default config
    config_logger.info("Loading packaged default config.")
    try:
        with importlib.resources.files('baberu.defaults').joinpath('default_config.yaml').open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError):
         config_logger.critical("Fatal: Could not find any configuration file, not even the packaged default.")
         raise RuntimeError("Fatal: Could not find any configuration file, not even the packaged default.")