from loguru import logger
from utils import load_config

cfg = load_config('config.yaml')


def get_logger(filename=None):
    if filename:
        logger.add(f"{cfg['log_folder']}/{filename}.log",
                   format="{time:DD-MM-YYYY HH:mm:ss} | {name} | {level} | {message}",
                   level="DEBUG")
    return logger
