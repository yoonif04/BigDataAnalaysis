import numpy as np


def create_add_number_data(total_count=100000):
    def _convert(numbers):
        a, b, c = numbers
        ca = _characterize(a)
        cb = _characterize(b)
        cc = _characterize(c)

        return [*ca, '+', *cb], cc

    def _characterize(integer):
        ch = "%d" % integer
        result = [x for x in ch]
        return result

    def _generate_tuple(a,b,c):
        _x, _y = _convert([a, b, c])
        leng = len(_x)
        _x.reverse()
        for _ in range(7 - leng):
            _x.append(' ')
        _x.reverse()
        leng = len(_y)
        _y.reverse()
        for _ in range(4 - leng):
            _y.append(' ')

        return _x, _y

    cX = []
    cY = []
    np.random.seed(0)
    for _ in range(total_count):
        a, b = np.random.randint(0, 999, 2)
        c = a+b
        x, y = _generate_tuple(a, b, c)
        cX.append(x)
        cY.append(y)
    cX = np.array(cX)
    cY = np.array(cY)

    ctX = []
    ctY = []
    np.random.seed(1)
    for _ in range(int(total_count/10)):
        a, b = np.random.randint(0, 999, 2)
        c = a+b
        x, y = _generate_tuple(a, b, c)
        ctX.append(x)
        ctY.append(y)
    ctX = np.array(ctX)
    ctY = np.array(ctY)

    return cX, cY, ctX, ctY