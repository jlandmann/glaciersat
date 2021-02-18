import pytest
from glaciersat.tests import requires_credentials
from glaciersat.utils import *
import configobj


@requires_credentials
def test_get_credentials():
    cred = get_credentials(credfile=None)
    assert isinstance(cred, configobj.ConfigObj)
    cred = get_credentials(credfile='.\\.credentials')
    assert isinstance(cred, configobj.ConfigObj)
