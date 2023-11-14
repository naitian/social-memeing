import numpy as np


def to_binary_array(num, num_chars=16):
    # https://stackoverflow.com/a/29091970
    return np.frombuffer(bin(num)[2:].zfill(num_chars * 4).encode(), "u1") - ord("0")


def to_int(string):
    if type(string) is not str and np.isnan(string):
        return 0
    return int(string, base=16)