import logging

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("HGG")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

def warning_on(condition, msg):
    if condition:
        logger.warning(msg)
    pass
    

def logger_activate_debug_mode():
    """change global logger"""
    logger.info("===== 'logger.debug()' enabled =====")
    logger.setLevel(logging.DEBUG)
    logger.handlers[0].setLevel(logging.DEBUG)
