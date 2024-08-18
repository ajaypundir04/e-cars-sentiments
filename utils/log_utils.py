import logging

class LoggerManager:
    """
    LoggerManager class to configure and manage loggers with different levels.
    """

    def __init__(self, level=logging.INFO):
        """
        Initialize the LoggerManager with a default logging level.

        :param level: Default logging level (e.g., logging.INFO, logging.DEBUG, logging.ERROR).
        """
        self.level = level

    def get_logger(self, name, level=None):
        """
        Get a logger with the specified name and level.

        :param name: Name of the logger, typically the class name or module name.
        :param level: Logging level (e.g., logging.INFO, logging.DEBUG, logging.ERROR). Defaults to the LoggerManager's level.
        :return: Configured logger instance.
        """
        if level is None:
            level = self.level

        # Configure the logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handler to the logger if not already added
        if not logger.handlers:
            logger.addHandler(ch)

        return logger
