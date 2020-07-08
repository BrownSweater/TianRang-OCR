import logging
from os import path, makedirs


class Logger(object):
    '''
    receive and save image from redis stream
    '''
    def __init__(self, input_log_path, name=None):

        logFormatterStr = "%(asctime)s - %(levelname)s - [%(filename)s: %(lineno)d] - %(message)s"
        logging.basicConfig(level = logging.INFO,format = logFormatterStr)
        if name is None:
            self._logger = logging.getLogger()
        else:
            self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        logging.getLogger("kafka").setLevel(logging.WARNING)

        input_log_dir = path.dirname(input_log_path)
        if input_log_dir and not path.exists(input_log_dir):
            makedirs(input_log_dir)

        fh = logging.FileHandler(input_log_path, mode='a+')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

        formatter = logging.Formatter(logFormatterStr)
        fh.setFormatter(formatter)

        self._logger.addHandler(fh)

    def logger(self):
        return self._logger

    # def debug(self, msg):
    #     self._logger.debug(msg)
    # def info(self, msg):
    #     self._logger.info(msg)
    # def warning(self, msg):
    #     self._logger.warning(msg)
    # def error(self, msg):
    #     self._logger.error(msg)
    # def fatal(self, msg):
    #     self.fatal(msg)
    # def critical(self, msg):
    #     self._logger.critical(msg)
    #     self._logger.handlers[0].flush()


# logger = Logger(cfg_log.LOG_DIR).logger()