from configobj import ConfigObj, ConfigObjError
import os
import sys

import logging
# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def get_credentials(credfile: str = None) -> ConfigObj:
    """
    Get credentials to use SentinelHub login.

    Parameters
    ----------
    credfile : str or None
        Credentials file in configObj style. Default: None (take the one under
        the top level directory with the name ".credentials".

    Returns
    -------
    cr: configobj.ConfigObj
        Configuration object with the credentials.
    """
    if credfile is None:
        credfile = os.path.join(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__))), '.credentials')

    try:
        cr = ConfigObj(credfile, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Credentials file could not be parsed (%s): %s',
                     credfile, e)
        sys.exit()

    return cr

