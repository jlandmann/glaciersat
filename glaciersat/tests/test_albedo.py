from glaciersat.core.albedo import *

b = np.array([[0.1, 0.2], [0.3, 0.4]])
g = np.array([[0.1, 0.2], [0.3, 0.4]])
r = np.array([[0.1, 0.2], [0.3, 0.4]])
nir = np.array([[0.1, 0.15], [0.2, 0.25]])
swir1 = np.array([[0.025, 0.05], [0.075, 0.1]])
swir2 = np.array([[0.025, 0.05], [0.075, 0.075]])


def test_get_broadband_albedo_knap():
    a = get_broadband_albedo_knap(g, nir)
    a_res = np.array([[0.07369, 0.1431425], [0.20906, 0.2714425]])
    np.testing.assert_allclose(a, a_res)

    # test negative
    np.testing.assert_equal(get_broadband_albedo_knap(0.0001, 0.01290877), 0.)


def test_get_broadband_albedo_liang():
    a = get_broadband_albedo_liang(b, r, nir, swir1, swir2)
    a_res = np.array([[0.088025, 0.1592], [0.230375, 0.29975]])
    np.testing.assert_allclose(a, a_res)

    # very improbable, but here we go:
    np.testing.assert_equal(get_broadband_albedo_liang(0.0001, 0.0001, 0.0001,
                                                       0.00001, 0.0001), 0.)


def test_get_broadband_albedo_bonafoni():
    a = get_broadband_albedo_bonafoni(b, g, r, nir, swir1, swir2)
    a_res = np.array([[0.08869, 0.160295], [0.2319, 0.30266]])
    np.testing.assert_allclose(a, a_res)


def test_get_proxy_albedo_mccarthy():
    a = get_proxy_albedo_mccarthy(r, g, b)
    a_res = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(a, a_res)


def test_get_ensemble_albedo():
    a = get_ensemble_albedo(b, g, r, nir, swir1, swir2)
    a_res = np.array([[0.07369, 0.1431425],
                      [0.20906, 0.2714425],
                      [0.088025, 0.1592],
                      [0.230375, 0.29975],
                      [0.08869, 0.160295],
                      [0.2319, 0.30266]])
    np.testing.assert_allclose(a, a_res)
