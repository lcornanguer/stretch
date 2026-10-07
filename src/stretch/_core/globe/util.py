
def is_insignificant(gain, alpha=0.05): # hypercompression test
    return gain < 0 or 2 ** (-gain) > alpha # gain must be over 4.3
