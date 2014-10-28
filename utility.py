from numpy import frombuffer, dtype, empty, asarray, iinfo, log10
from scipy.signal import lfilter, freqz
from PySide.QtCore import QPoint

def byteToPCM(data, sample_width):
    d_type = 'float'
    if sample_width == 2:
        d_type = 'short'
    return frombuffer(data, dtype = d_type)

def pcmToFloat(sig, type='float32'):
    sig = asarray(sig)
    if sig.dtype.kind != 'i':
        raise TypeError('signal must be integer')
    type = dtype(type)
    if type.kind != 'f':
        raise TypeError('type must be float')

    return sig.astype(type) / type.type(-iinfo(sig.dtype).min)

def floatToPCM(sig, dtype='int16'):
    return (sig * iinfo(dtype).max).astype(dtype)

def sosfilter(sos, zi_in, x):
    y = x
    zi_out = zi_in
    for i in range(len(sos)):
        y, zi_out[i] = lfilter(sos[i,:3], sos[i,3:], y, zi = zi_in[i])
    return y, zi_out

def sosfreqz(sos, ws = None):
    if ws is None:
        H = [1] * 512        
    else:
        H = [1] * len(ws)

    for i in range(len(sos)):
        w, h = freqz(sos[i,:3], sos[i, 3:], worN = ws)
        H *= h
    return w, H

def toPixelCords(width, height, x, xaxis, y = 0, yaxis = None):
    xmin = xaxis.min
    xmax = xaxis.max       

    if xaxis.log:
        xp = log10(x / xmin + 0.000001) / log10(xmax / xmin) * width
    else:
        xp = (x - xmin) / (xmax - xmin) * width
    if yaxis != None:
        ymin = yaxis.min
        ymax = yaxis.max
        yp = (y - ymax) / (ymin - ymax) * height
        return QPoint(xp, yp)
    else:
        return xp

def fromPixelCords(width, height, point, xaxis, yaxis):
    xmin = xaxis.min
    xmax = xaxis.max
    ymin = yaxis.min
    ymax = yaxis.max

    xp = point.x()
    yp = point.y()

    if xaxis.log:
        x = 10 ** (xp * log10(xmax / xmin) / width + log10(xmin))
    else:
        x = xp * (xmax - xmin) / width + xmin
    y = yp * (ymin - ymax) / height + ymax
    return x, y