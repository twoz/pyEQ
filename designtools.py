from numpy import asarray, array, append, zeros, ones, prod

def cplxpair(x, tol=1e-12) :
    """
    Sort the numbers z into complex conjugate pairs ordered by
    increasing real part.  With identical real parts, order by
    increasing imaginary magnitude. Place the negative imaginary
    complex number first within each pair. Place all the real numbers
    after all the complex pairs (those with `abs (z.imag / z) < tol)',
    where the default value of TOL is `100 * EPS'.
    
    Inputs :
        x : input array.
        tol : toleranze of differenz of complex number and his conjugate pair.
        
    Outputs : y
        y : Complex conjugate pair
    """
    
    if sum(x.shape) == 0 : return x
    if not 'complex' in str(x.dtype) : return x
    
    # Reshape input to 1D array to simplify algorithm
    x_shape = x.shape
    x  = x.reshape(prod(array(list(x_shape))))
    xout = array([])
    tol = abs(tol)
    
    # Save original class of input
    x_orig_class = x[0].__class__
    
    # New rule to sort
    class __cplxpairsort__ (x_orig_class) :
        def __gt__(self, a) :
            return self.real > a.real
        def __ge__(self, a) :
            return self.real > a.real or self.__eq__(a)
        def __lt__(self, a) :
            return self.real < a.real
        def __le__(self, a) :
            return self.real < a.real or self.__eq__(a)
        def __eq__(self, a) :
            return abs(self.real-a.real) <= tol and abs(self.imag+a.imag) <= tol
        def __ne__(self, a) :
            return not self.__eq__(a)
    
    def post_sort(x_sort):
        i = 0
        pair = []
        nopair = []
        while True :
            re, im = x_sort[i].real - x_sort[i+1].real,x_sort[i].imag + x_sort[i+1].imag
            if abs(re) <= tol and abs(im) <= tol :
                pair.append(x_sort[i])
                pair.append(x_sort[i+1])
                i += 1
            else :
                nopair.append(x_sort[i])
            i += 1
            if i >= len(x_sort)-1 : break
        if len(pair) + len(nopair) != len(x_sort) : nopair.append(x_sort[-1])
        return append(pair, nopair)
    
    # Change dtype of input to pair
    x = x.astype(__cplxpairsort__)
    
    # Do it like multi-demension array with array sclicing.
    for i in range((int)(x.shape[0]/x_shape[-1])):
        x_sort = 1*x[i*x_shape[-1] : (i+1)*x_shape[-1]]
        x_sort.sort()
        x_sort = post_sort(x_sort)
        xout = append(xout, 1*x_sort)
    
    # Return with original shape and original class
    return xout.reshape(x_shape).astype(x_orig_class)

def cplxreal(z, tol=1e-12) :
    """
    Split the vector z into its complex (zc) and real (zr) elements,
    eliminating one of each complex-conjugate pair.

    Inputs:
        z   : row- or column-vector of complex numbers
        tol : tolerance threshold for numerical comparisons

    Ouputs:
        zc : elements of z having positive imaginary parts
        zr : elements of z having zero imaginary part

    Each complex element of Z is assumed to have a complex-conjugate
    counterpart elsewhere in Z as well.  Elements are declared real if
    their imaginary parts have magnitude less than tol.
    
    Note : This function is modified from signal package for octave 
    (http://octave.sourceforge.net/signal/index.html)
    """
    if z.shape[0] == 0 : zc=[];zr=[]
    else :
        zcp = cplxpair(z)
        nz  = len(z)
        nzrsec = 0
        i = nz
        while i and abs(zcp[i-1].imag) < tol:
            zcp[i-1] = zcp[i-1].real
            nzrsec = nzrsec+1
            i=i-1
        
        nzsect2 = nz-nzrsec
        if nzsect2%2 != 0 :
            raise ValueError('cplxreal: Odd number of complex values!')
    
        nzsec = nzsect2/2
        zc = zcp[1:nzsect2:2]
        zr = zcp[nzsect2:nz]
    return asarray(zc), asarray(zr)

def zpk2sos(z,p,k) :
    """
    Convert filter poles and zeros to second-order sections.
    
    Inputs:
        z : column-vector containing the filter zeros
        p : column-vector containing the filter poles
        k : overall filter gain factor
    
    Outputs:
        sos : matrix of series second-order sections, one per row:
        k : is an overall gain factor that effectively scales any
        one of the Bi vectors.
    
    Example:
        z,p,k = tf2zpk([1, 0, 0, 0, 0, 1],[1, 0, 0, 0, 0, .9])
        sos,k = zp2sos(z,p,k)
    
        sos =
            1.0000    0.6180    1.0000    1.0000    0.6051    0.9587
            1.0000   -1.6180    1.0000    1.0000   -1.5843    0.9587
            1.0000    1.0000         0    1.0000    0.9791         0
    
        k =
            1
    
    See also: tf2sos zpk2tf tf2zpk sos2zpk sos2tf.
    
    Note : This function is modified from signal package for octave 
    (http://octave.sourceforge.net/signal/index.html)
    """
    zc,zr = cplxreal(array(z))
    pc,pr = cplxreal(array(p))
    
    nzc = len(zc)
    npc = len(pc)
    nzr = len(zr)
    npr = len(pr)
    
    # Pair up real zeros:
    if nzr :
        if nzr%2 == 1 : zr = append(zr,0); nzr=nzr+1
        nzrsec = nzr/2.0
        zrms = -zr[:nzr-1:2]-zr[1:nzr:2]
        zrp  =  zr[:nzr-1:2]*zr[1:nzr:2]
    else :
        nzrsec = 0
    
    # Pair up real poles:
    if npr :
        if npr%2 == 1 : pr = append(pr,0); npr=npr+1
        nprsec = npr/2.0
        prms = -pr[:npr-1:2]-pr[1:npr:2]
        prp  =  pr[:npr-1:2]*pr[1:npr:2]
    else :
        nprsec = 0
    
    nsecs = max(nzc+nzrsec,npc+nprsec)
    
    # Convert complex zeros and poles to real 2nd-order section form:
    zcm2r = -2*zc.real
    zca2  = abs(zc)**2
    pcm2r = -2*pc.real
    pca2  = abs(pc)**2
    
    sos = zeros((nsecs,6))
    
    # all 2nd-order polynomials are monic
    sos[:,0] = ones(nsecs)
    sos[:,3] = ones(nsecs)
    nzrl = nzc+nzrsec # index of last real zero section
    nprl = npc+nprsec # index of last real pole section
    
    for i in range(int(nsecs)) :
        if   i+1 <= nzc : # lay down a complex zero pair:
            sos[i,1:3] = append(zcm2r[i], zca2[i])
        elif i+1 <= nzrl: #lay down a pair of real zeros:
            sos[i,1:3] = append(zrms[i-nzc], zrp[i-nzc])
        if   i+1 <= npc : # lay down a complex pole pair:
            sos[i,4:6] = append(pcm2r[i], pca2[i])
        elif i+1 <= nprl: # lay down a pair of real poles:
            sos[i,4:6] = append(prms[i-npc], prp[i-npc])
    
    if len(sos.shape) == 1 : sos = array([sos])

    sos[0,0:3] *= k
     
    return sos, k