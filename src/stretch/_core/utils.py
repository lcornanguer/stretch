import math

def universal_integer_encoding(i, c=2.865064):  # non-negative only
    bits = math.log2(c)
    while i > 1:
        i = math.log2(i)
        bits += i
    return bits


def universal_real_encoding(z, precision):
    if z == 0:
        return 0
        # return universal_integer_encoding(1)

    sign_cost = 1 if z < 0 else 0
    z = abs(z)

    s = round((10 ** precision) * z)

    return sign_cost + universal_integer_encoding(s)
