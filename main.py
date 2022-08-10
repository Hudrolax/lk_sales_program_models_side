from data_worker import DataWorker
import logging
import argparse
import sys


def create_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-p', '--production', default='true')
    _parser.add_argument('-rh', '--redis_host', default='192.168.19.18')
    _parser.add_argument('-rdb', '--redis_db', default='2')
    return _parser


parser = create_parser()
params = parser.parse_args(sys.argv[1:])


PRODUCTION = params.production.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

print(f'production {PRODUCTION}')
print(f'redis_host {params.redis_host}')
print(f'redis_db {int(params.redis_db)}')

WRITE_LOG_TO_FILE = False
LOG_FORMAT = '%(name)s (%(levelname)s) %(asctime)s: %(message)s'
if not params.production:
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

logger = logging.getLogger('main')

if WRITE_LOG_TO_FILE:
    logging.basicConfig(filename='lk_sales_program_models_side.txt', filemode='w', format=LOG_FORMAT, level=LOG_LEVEL,
                        datefmt='%d.%m.%y %H:%M:%S')
else:
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, datefmt='%d.%m.%y %H:%M:%S')

logging.getLogger('werkzeug').setLevel(logging.WARNING)

if __name__ == '__main__':
    dw = DataWorker(production=bool(PRODUCTION), redis_host=str(params.redis_host), redis_db=int(params.redis_db))
    dw.run()
