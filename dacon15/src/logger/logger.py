import logging
import logging.config
import json
import os

class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, prefix, logger):
        super(LoggerAdapter, self).__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return '[%s] %s' % (self.prefix, msg), kwargs

if __name__ == '__main__':

    with open('logging.json', 'rt') as f:
        config = json.load(f)

    logging.config.dictConfig(config)

    logger = logging.getLogger("")

    logger = LoggerAdapter("SAS", logger)
    logger.info("test!!!")

