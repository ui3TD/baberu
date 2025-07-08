from baberu.tools import file_utils
from baberu.constants import APP_NAME

import platformdirs
import yaml

import shutil
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

    # 3. Check user-specific config directory. If not found, create from default.
    user_config_dir = Path(platformdirs.user_config_dir(APP_NAME))
    user_config_path = user_config_dir / "config.yaml"

    if not user_config_path.exists():
        config_logger.info(f"User config not found. Creating default at {user_config_path.resolve()}")
        try:
            # Find the packaged default config to use as a template
            with importlib.resources.files('baberu.defaults').joinpath('default_config.yaml') as default_path:
                # Ensure the destination directory exists and copy the file
                user_config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(default_path, user_config_path)
        except (FileNotFoundError, PermissionError) as e:
            config_logger.warning(
                f"Could not create user config file ({e}). "
                "Loading packaged default as a temporary fallback."
            )
            # 4. Fallback to loading the packaged default directly
            try:
                with importlib.resources.files('baberu.defaults').joinpath('default_config.yaml').open('r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except (FileNotFoundError, ModuleNotFoundError):
                config_logger.critical("Fatal: Could not find the packaged default config.")
                raise RuntimeError("Fatal: Could not find the packaged default config.")

    # Load the user config (either pre-existing or newly created)
    config_logger.info(f"Loading config from user directory: {user_config_path}")
    with open(user_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)