
from environment import *


# calcuate the time when the tumor is diagnosed
def cal_time(x_,tumor):
    for i in range(len(x_.T[0])):
        if x_.T[0][i]+x_.T[1][i]>tumor:
            break
    return i

