import yaml

from src.schemas.enums.config_paths import ConfigName
from src.schemas.types import ProjectSettings


def load_config(config: ConfigName = ConfigName.DEFAULT) -> ProjectSettings:
    with open(config.path(), "r") as f:
        raw = yaml.safe_load(f)
    return ProjectSettings(**raw)
