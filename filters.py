import scipy.signal as scsig
import numpy as np
from designtools import zpk2sos
from utility import sosfilter, sosfreqz

# Typical IIR _filters found in parametric equalizers nowadays
# LPButter & HPButter are Butterworth _filters of order 2,4 or 8
# Brickwall are eliptic filters
# Peak & shelving _filters are second-order resonant filters with adjustable Q-factor
class FilterType:

     LPButter = 0
     LPBrickwall = 1
     HPButter = 2
     HPBrickwall = 3
     LShelving = 4
     HShelving = 5
     Peak = 6

# Constructor designs a filter
# elliptic & butter filters are designed as zero-poles and broken into
# cascaded biquads (second-order-state) to avoid numerical errors
class Filter:
    def __init__(self, type, fc, gain = 0, Q = 1, enabled = True):
        self._enabled = enabled
        self._type = type
        self._fc = fc
        self._g = gain
        self._Q = Q

        if type == FilterType.HPBrickwall:
            z, p, k = scsig.ellip(12, 0.01, 80, fc, 'high', output='zpk')
            self._sos = zpk2sos(z, p, k)[0]
        elif type == FilterType.LPBrickwall:
            z, p, k = scsig.ellip(12, 0.01, 80, fc, 'low', output='zpk')
            self._sos = zpk2sos(z, p, k)[0]
        elif type == FilterType.HPButter:
            z, p, k = scsig.butter(2 ** Q, fc, btype = 'high', output='zpk')
            self._sos = zpk2sos(z, p, k)[0]
        elif type == FilterType.LPButter:
            z, p, k = scsig.butter(2 ** Q, fc, output='zpk')
            self._sos = zpk2sos(z, p, k)[0]
        elif type == FilterType.LShelving or type == FilterType.HShelving:
            A = 10 ** (gain / 20)
            wc = np.pi * fc
            wS = np.sin(wc)
            wC = np.cos(wc)
            alpha = wS / (2 * Q)
            beta = A ** 0.5 / Q
            c = 1
            if type == FilterType.LShelving:
                c = -1

            b0 = A * (A + 1 + c * (A - 1) * wC + beta * wS)
            b1 = - c * 2 * A * (A - 1 + c * (A + 1) * wC)
            b2 = A * (A + 1 + c * (A - 1) * wC - beta * wS)
            a0 = (A + 1 - c * (A - 1) * wC + beta * wS)
            a1 = c * 2 * (A - 1 - c * (A + 1) * wC)
            a2 = (A + 1 - c * (A - 1) * wC - beta * wS)
            self._sos = np.array([[ b0, b1, b2, a0, a1, a2 ]])
        elif type == FilterType.Peak:
            self.g = gain
            wc = np.pi * fc
            b, a = scsig.bilinear([1, 10 ** (gain / 20) * wc / Q, wc ** 2],
                [1, wc / Q, wc ** 2])
            self._sos = np.append(b, a).reshape(1, 6)

        self._ord = self._sos.shape[0] * 2
        self.icReset()

    def icReset(self):
        self._zi = np.zeros(shape = (self._sos.shape[0], 2))

# Class representing a cascade of filters
# Currently there is 5 user adjustable filters
# Filters can be enabled/disabled or changed at any time
class FilterChain:
    def __init__(self):
        self._filters = []

    def sos(self, i = -1):
        """
        Returns second-order-section matrix of this chain or one filt.
        """
        if i != -1:
            return self._filters[i]._sos
        
        sos = np.ones(shape = (1,6))

        for filt in self._filters:
            if filt._enabled is True:
                sos = np.append(sos, filt._sos, axis = 0)
        return sos

    def setFiltEnabled(self, i, enable):
        filt = self._filters[i]
        filt._enabled = enable
        if enable is True:
            filt._zi = np.zeros(shape = (filt._sos.shape[0], 2))

    def updateFilt(self, i, new):
        old = self._filters[i]
        self._filters[i] = new
        if old._type == new._type and old._ord == new._ord:
            self._filters[i]._zi = old._zi

    def getZi(self):
        zi = [[0, 0]]
        for filt in self._filters:
            if filt._enabled is True:
                zi.extend(filt._zi)
        return zi

    def updateZi(self, zi):
        n = 1
        for filt in self._filters:
            if filt._enabled is True:
                m = filt._sos.shape[0]
                filt._zi = zi[n:n+m]
                n += m

    def reset(self):
        for filt in self._filters:
            filt.icReset()
                    
    def filter(self, x):
        y, zi = sosfilter(self.sos(), self.getZi(), x)
        self.updateZi(zi)
        return y