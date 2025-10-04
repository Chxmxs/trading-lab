import logging
import os

def setup_logging(log_file='explorer.log'):
    """
    Configure a file logger for the Strategy Explorer.
    Creates a 'logs' directory alongside the script if it doesn't exist.
    """
    log_path = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)
