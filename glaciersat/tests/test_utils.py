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


def test_declutter():
    input = np.zeros((7, 7), dtype=int)
    input[1:6, 1:6] = 1

    # erode a lot, then leave as is
    res = declutter(input, 3, 1).astype(int)
    desired = np.zeros((7, 7), dtype=int)
    desired[2:5, 2:5] = 1
    np.testing.assert_array_equal(res, desired)

    # do not erode, but dilate
    res = declutter(input, 1, 3).astype(int)
    desired = np.ones((7, 7), dtype=int)
    np.testing.assert_array_equal(res, desired)

    # erode and dilate (the offset is unfortunate though)
    res = declutter(input, 3, 2).astype(int)
    desired = np.zeros((7, 7), dtype=int)
    desired[1:5, 1:5] = 1
    np.testing.assert_array_equal(res, desired)

    res = declutter(input, 2, 3).astype(int)
    desired = np.zeros((7, 7), dtype=int)
    desired[1:, 1:] = 1
    np.testing.assert_array_equal(res, desired)

