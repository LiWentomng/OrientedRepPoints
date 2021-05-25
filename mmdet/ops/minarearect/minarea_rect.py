import torch
from . import minarearect

def minaerarect(pred):
    rbbox = minarearect.minareabbox(pred)
    rbbox = rbbox.reshape(-1, 8)
    return rbbox