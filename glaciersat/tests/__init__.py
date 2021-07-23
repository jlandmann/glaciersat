import os
from configobj import ConfigObj, ConfigObjError
import pytest


# check if there is a credentials file (should be added to .gitignore)
cred_path = os.path.abspath(os.path.join(__file__, "../../..", '.credentials'))
HAS_CREDENTIALS = False
if os.path.exists(cred_path):
    HAS_CREDENTIALS = True
    try:
        cred = ConfigObj(cred_path)
    except ConfigObjError:
        raise


def requires_credentials(test):
    """Test decorator to make it require login credentials."""
    msg = 'This test requires credentials'
    return test if HAS_CREDENTIALS else pytest.mark.skip(msg)
