import numpy as np

from corneto.extensions import numba


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _loop(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r


@numba.vectorize(["float64(float64, float64)"], target="parallel")
def vec_sum(x, y):
    return x + y


@numba.guvectorize(["void(float64[:], intp[:], float64[:])"], "(n),()->(n)")
def move_mean(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count


def test_guvectorize_numba():
    arr = np.arange(20, dtype=np.float64).reshape(2, 10)
    result = move_mean(arr, 3)
    expected = np.array(
        [
            [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_jit_loop_numba():
    _loop(np.ones(10000))
    assert True


def test_vec_sum_numba():
    vec_sum(np.ones(10000), np.ones(10000))
    assert True
