import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - %(lineno)d',
    datefmt='%m-%d-%Y %H:%M:%S'
)