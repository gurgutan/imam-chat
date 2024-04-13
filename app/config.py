import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from logger import logger
from app.utils import merge

DEFAULT_CONFIG_FILENAME = "config.yml"


def _get_config(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / DEFAULT_CONFIG_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        logger.info(f"Использую файл конфигурации {path}")
        return yaml.safe_load(f)


def get_config(path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    default_config = _get_config(Path(__file__).parent)
    if path is None:
        path = Path() / DEFAULT_CONFIG_FILENAME
        if not path.is_file():
            return default_config
    config = _get_config(path)
    return merge(default_config, config)
