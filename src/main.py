import logging
import datetime

if __name__ == '__main__':
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = "../logs/" + ct + "_run.log"

    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s',
        filemode='w',
        level=logging.DEBUG)
