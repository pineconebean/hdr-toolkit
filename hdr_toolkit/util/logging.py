import logging


def get_logger(name, output_file, formatter=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create handlers for logger
    file_handler = logging.FileHandler(output_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # create a formatter for handlers
    if formatter is None:
        formatter = logging.Formatter('%(asctime)s;%(message)s', '%Y-%m-%d %H:%M')
    else:
        formatter = logging.Formatter(formatter)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
