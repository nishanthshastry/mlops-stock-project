import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_logger(name: str):
    setup_logging()
    return logging.getLogger(name)
