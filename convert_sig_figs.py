#The following constant was computed in maxima 5.35.1 using 64 bigfloat digits of precision
__logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1

import numpy as np

def RoundToSigFigs( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or np.issubdtype(sigfigs, np.integer)):
        raise TypeError( "RoundToSigFigs: sigfigs must be an integer." )

    if not np.all(np.isreal( x )):
        raise TypeError( "RoundToSigFigs: all x must be real." )

    if sigfigs <= 0:
        raise ValueError( "RoundtoSigFigs: sigfigs must be positive." )

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = np.frexp( absx )

    decimalExponents = __logBase10of2 * binaryExponents
    intParts = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - intParts)

    if type(mantissas) is float or np.issctype(np.dtype(mantissas)):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0

    elif np.issubdtype(mantissas, np.ndarray):
        fixmsk = mantissas < 1.0
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    return xsgn * np.around( mantissas, decimals=sigfigs - 1 ) * 10.0**intParts