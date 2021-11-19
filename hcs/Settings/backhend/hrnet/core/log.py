import logging


def create_logger(file):
    """Creates logger.
    Message whose level is at least logging.INFO will be printed.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    fstream = logging.FileHandler(file)
    fstream.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    fstream.setFormatter(formatter)
    logger.addHandler(fstream)

    return logger


if __name__ == '__main__':
    logger = create_logger('log.txt')
    logger.info('hello')
    logger.debug('bebe')