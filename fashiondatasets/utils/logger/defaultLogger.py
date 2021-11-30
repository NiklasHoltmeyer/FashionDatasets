import logging
import sys

logger_names = [
    "fashiondataset",
    "fashion_pair_gen",
    "fashion_evaluate",
    "fashion_image_preprocess"
]

#verbose_logger = [
#    "fashiondataset",
#    "FashionNet",
#    "fashiondataset_time_logger"
#]

# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,PyArgumentList
def defaultLogger(name="fashiondataset", level=logging.DEBUG, handlers=None,
                  format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'):
    handlers = handlers if handlers else [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(level=level, format=format, handlers=handlers)

    logger = logging.getLogger(name)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("nltk_data").setLevel(logging.WARNING)
    logging.getLogger("pysndfx").setLevel(logging.WARNING)
    logging.getLogger('selenium.webdriver.remote.remote_connection').setLevel(logging.WARNING)
    logging.getLogger('connectionpool').setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger

