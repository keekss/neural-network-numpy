import logging

log_format = (
    '%(funcName)s(), line %(lineno)d (%(asctime)s)\n'
    '-> [%(levelname)s] %(message)s\n'
)

date_format ='%Y-%m-%d @ %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format
)