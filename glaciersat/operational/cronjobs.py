# Built-ins
from collections import OrderedDict
import datetime as dt
import warnings

# Externals
from sentinelsat import SentinelAPI
import pandas as pd
import schedule

# Locals
from glaciersat import utils, cfg

# Module Logger
import logging

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def check_for_new_scenes():
     # 1 - check
     # 2 - download
     raise NotImplementedError


def daily_tasks():

    log.info('Downloading scenes from yesterday...')
    today = pd.Timestamp.now().date()
    yesterday = today - pd.Timedelta(days=1)
    tiles = cfg.PARAMS['sentinel2_tiles_ch']

    # default Download is Sentinel-2
    p, d, t, f = utils.download_sentinel_tiles(yesterday, today, tiles)
    log.info('{} products searched: {},\n {} downloaded: {},\n {} triggered: '
             '{},\n {} failed: {}'.format(len(p),
                                          [v['title'] for k, v in p.items()],
                                          len(d),
                                          [v['title'] for k, v in d.items()],
                                          len(t),
                                          [v['title'] for k, v in t.items()],
                                          len(f),
                                          [v['title'] for k, v in f.items()]))

    # don't ask and just unzip all in the subdirs
    utils.unzip_sentinel()

    # distribute_to_domains
    # process albedo
    # process snow lines


if __name__ == '__main__':
    cfg.initialize()
    daily_tasks()
    schedule.every().day.at('10:00').do(daily_tasks)



