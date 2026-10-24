"""Stable semicircle moments for receiver quadrature."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def window_moments(low, high):
    """Integrals of sqrt(1-h²), h sqrt(1-h²), and 1-h² on a clipped window."""
    low, high = max(-1., low), min(1., high)
    if low >= high:
        return 0., 0., 0.
    width = high-low
    ca = math.sqrt(max(0., (1-low)*(1+low)))
    cb = math.sqrt(max(0., (1-high)*(1+high)))
    if ca+cb == 0.:
        return math.pi/2, 0., 4/3
    cosine = low*high+ca*cb
    sine = (high*ca-low*cb if cosine < 0.
            else width*(1+cosine)/(ca+cb))
    angle = math.atan2(sine, cosine)
    if angle < .5:
        term = angle**3/6
        minus = term
        for j in range(2, 12):
            term *= -angle*angle/((2*j)*(2*j+1))
            minus += term
            if abs(term) <= np.finfo(np.float64).eps*abs(minus):
                break
    else:
        minus = angle-sine
    # cos²((asin(low)+asin(high))/2), without inverse trigonometric
    # functions or cancellation near either grazing endpoint.
    mid_cos2 = ((1+ca*cb-low*high)/2 if cosine < 0.
                else (ca+cb)**2/(2*(1+cosine)))
    m0 = minus/2+mid_cos2*sine
    m1 = width*(low+high)*(ca*ca+ca*cb+cb*cb)/(3*(ca+cb))
    if low >= 0.:
        m2 = width*((1-low)*(1+low)-width*(low+width/3))
    elif high <= 0.:
        m2 = width*((1-high)*(1+high)+width*(high-width/3))
    else:
        m2 = width*(1-(low*low+low*high+high*high)/3)
    return m0, m1, m2



@njit(cache=True)
def root_moments(a,b):
    """Means of sqrt(1-h²) and h sqrt(1-h²) over [a,b]."""
    if a == b:
        root=math.sqrt(max(0., (1-a)*(1+a)))
        return root, a*root
    m0,m1,_=window_moments(a,b)
    return m0/(b-a),m1/(b-a)
