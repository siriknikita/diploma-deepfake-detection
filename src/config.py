import yaml

from src.schemas.custom_types import ProjectSettings
from src.schemas.enums.config_paths import ConfigName


def load_config(config: ConfigName = ConfigName.DEFAULT) -> ProjectSettings:
    """
    Load the project configuration from a YAML file.

    Args:
        config (ConfigName): The configuration file to load. Defaults to ConfigName.DEFAULT.
    Returns:
        ProjectSettings: The loaded project settings.
    """
    with open(config.path(), "r") as f:
        raw = yaml.safe_load(f)
    return ProjectSettings(**raw)
