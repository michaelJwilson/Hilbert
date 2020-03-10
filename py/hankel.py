import  numpy  as np
import  pylab  as pl

from    mcfit  import kernels, mcfit, transforms


x      = np.logspace(-3., 3., num=60, endpoint=False)

F      = 1. / (1. + x*x)**1.5

H      = mcfit(x, kernels.Mellin_BesselJ(0), 0.0)

y, G   = H(x**2 * F)

Gexact = np.exp(-y)

y, G   = transforms.Hankel(x)(F)

print H.check(x**2 * F)

print np.allclose(G, Gexact)

print y, G

