from baberu.constants import CONFIG_FILE_NAME
from baberu.errors import BaberuError
from baberu.settings import PROFILE_SECTIONS, Settings
from baberu.constants import TEMPLATE_CONFIG_FILE_NAME
from baberu.tools import file_utils
from baberu.constants import APP_NAME

import platformdirs
import yaml

import shutil
import importlib.resources
import logging
from pathlib import Path
from typing import Any

def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Overlay wins; nested tables merge instead of replacing each other."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_profiles(settings: Settings, lang: str) -> Settings:
    """Lays every matching profile over the settings, in the order they are listed."""
    model = settings.transcription.model
    matching = [profile for profile in settings.profiles if profile.matches(model, lang)]
    if not matching:
        return settings

    data = settings.model_dump()
    for profile in matching:
        unknown = set(profile.settings) - set(PROFILE_SECTIONS)
        if unknown:
            raise BaberuError(f"A profile can't change {', '.join(sorted(unknown))}. "
                              f"It can change: {', '.join(PROFILE_SECTIONS)}.")
        data = _merge(data, profile.settings)
        logging.getLogger(__name__).debug(f"Using the profile for model '{profile.model}' language '{profile.lang}'")
    return Settings.model_validate(data)


def config_dir(arg: Path | None = None) -> Path:
    """The folder of the config file load_config reads, for files kept beside it."""
    if arg and arg.exists():
        return arg.resolve().parent
    try:
        project_root = file_utils.get_project_dir()
        if (project_root / CONFIG_FILE_NAME).exists():
            return project_root
    except FileNotFoundError:
        pass
    return Path(platformdirs.user_config_dir(APP_NAME))


def load_config(arg: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file with a fallback mechanism.

    The lookup order is:
    1. Path specified by the commandline argument (not implemented).
    2. `config.yaml` in the project root (for development).
    3. `config.yaml` in the user's config directory (e.g., ~/.config/baberu/).
    4. The default `default_config.yaml` packaged with the application.
    """

    config_logger = logging.getLogger(__name__)

    # 1. Check arg
    if arg and arg.exists():
        with open(arg, 'r') as f:
            return yaml.safe_load(f)

    # 2. Check project root (for local development)
    try:
        project_root = file_utils.get_project_dir()
        dev_config_path = project_root / CONFIG_FILE_NAME
        if dev_config_path.exists():
            config_logger.debug(f"Loading config from dev directory: {dev_config_path}")
            with open(dev_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    except FileNotFoundError:
        pass

    # 3. Check user-specific config directory. If not found, create from default.
    user_config_dir = Path(platformdirs.user_config_dir(APP_NAME))
    user_config_path = user_config_dir / CONFIG_FILE_NAME

    if not user_config_path.exists():
        config_logger.info(f"User config not found. Creating default at {user_config_path.resolve()}")
        try:
            # Find the packaged default config to use as a template
            with importlib.resources.as_file(importlib.resources.files('baberu.defaults').joinpath(TEMPLATE_CONFIG_FILE_NAME)) as default_path:
                # Ensure the destination directory exists and copy the file
                user_config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(default_path, user_config_path)
        except (FileNotFoundError, PermissionError) as e:
            config_logger.warning(
                f"Could not create user config file ({e}). Loading packaged default as a temporary fallback."
            )
            # 4. Fallback to loading the packaged default directly
            try:
                with importlib.resources.files('baberu.defaults').joinpath(TEMPLATE_CONFIG_FILE_NAME).open('r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except (FileNotFoundError, ModuleNotFoundError):
                config_logger.critical("Fatal: Could not find the packaged default config.")
                raise RuntimeError("Fatal: Could not find the packaged default config.")

    # Load the user config (either pre-existing or newly created)
    config_logger.debug(f"Loading config from user directory: {user_config_path}")
    with open(user_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)