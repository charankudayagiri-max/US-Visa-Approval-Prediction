from us_visa.logger import logging
# logging.info("welcome to our logging")
from us_visa.exception import USvisaException
import sys

try:
    x = 1/0
except Exception as e:
    logging.error(USvisaException(e,sys))