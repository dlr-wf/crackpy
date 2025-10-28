import logging.config
import yaml
from importlib import resources

def setup_logging(default_level=logging.INFO):
    """Load logging configuration from packaged YAML file."""
    try:
        with resources.files("crackpy").joinpath("logging.yaml").open("r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).warning("Falling back to basic logging configuration: %s", e)
