import logging

FORMAT = "[GPTVM] %(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("gptvm")


def setDebug():
    logger.setLevel(logging.DEBUG)


def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)
    logger.critical(msg, *args, **kwargs)
