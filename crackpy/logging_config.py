import logging.config
import os
from datetime import datetime
from importlib import resources
from pathlib import Path

import yaml


def _prepare_file_handlers(cfg: dict, appname: str = "crackpy"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()

    for h in cfg.get("handlers", {}).values():
        if "filename" not in h:
            continue

        base = Path(os.path.expandvars(os.path.expanduser(h["filename"])))
        if not base.is_absolute():
            base = Path.home() / ".cache" / appname / "logs" / base

        filename = f"{base.stem}_{timestamp}_{pid}{base.suffix or '.log'}"
        base = base.with_name(filename)

        base.parent.mkdir(parents=True, exist_ok=True)
        h["filename"] = str(base)


def setup_logging(default_level=logging.INFO):
    try:
        with resources.files("crackpy").joinpath("logging.yaml").open("r") as f:
            cfg = yaml.safe_load(f)
        _prepare_file_handlers(cfg)
        logging.config.dictConfig(cfg)
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.warning("Falling back to basic logging: %s", e)
