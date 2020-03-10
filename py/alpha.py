import  matplotlib;         matplotlib.use('PDF')

import  astropy
import  scipy
import  scipy.ndimage
import  numpy               as     np
import  pylab               as     pl
import  matplotlib.pyplot   as     plt

from    astropy.cosmology   import FlatLambdaCDM
from    astropy.cosmology   import Planck15
from    astropy             import constants as const
from    scipy.misc          import derivative

plt.style.use('ggplot')


## per sq deg.
data  = np.loadtxt("nz_BGS_bright_decals.txt")
lozs  = data[:,0]
hizs  = data[:,1]
dNdz  = data[:,2]

dzs   = hizs - lozs
midzs = lozs + 0.5 * dzs 

dNdz  =  dNdz[midzs <= 0.21]
midzs = midzs[midzs <= 0.21]

## check constant dz.
## print np.unique(dzs)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

## H(z) = H0 * E(z)
ez    = cosmo.efunc(midzs)
Hz    = cosmo.H0 * ez
Dc    = cosmo.comoving_distance(midzs) ## [Mpc]

Honc  = Hz / const.c.to('km/s')

nbar  = dNdz * Honc / Dc**2.
nbar *= (180. / np.pi)**2.             ## [Mpc ^ -3]

h     = cosmo.H0.value / 100.

Dc   *= h                              ## [Mpc / h]
nbar /= h**3.                          ## [(Mpc / h) ^ -3]

## smooth 
smooth = scipy.ndimage.gaussian_filter(np.log(Dc.value * Dc.value * nbar.value), 1, order=0, mode='reflect', cval=0.0, truncate=4.0)

## polynomial fit
order  = 5

fit          = np.polyfit(Dc.value, smooth, order)
poly_approx  =  np.poly1d(fit)

pl.plot(Dc, np.log(Dc.value * Dc.value * nbar.value), label='BGS')
pl.plot(Dc,                                   smooth, label='smooth')
pl.plot(Dc, poly_approx(Dc),                          label= str(order) + 'th order fit')

pl.xlabel(r'$\chi$')
pl.ylabel(r'$\log | \chi^2 \ \bar n(\chi)|$')

ax    = plt.gca()

ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

pl.legend(loc=2)

pl.savefig('BGS_nx.pdf', bbox_inches='tight')
  

## alpha plot.
pl.clf()

## local power law
## nbar = 1. / Dc ** 3.  

alpha   = nbar * Dc**2.  
alpha   = alpha.value    ## units? ln argument should be dimensionless; see eqn. (3) of Castorina and White.

alpha   = np.log(alpha)

## d / dln|X| = X * (d / dX) 

alpha   = Dc * np.gradient(alpha, Dc)

pl.plot(Dc,                 alpha, label=r'BGS')
pl.plot(Dc, 2. * np.ones_like(Dc), label=r'')

## poly approx 
zs  = np.linspace(0.005, 0.2, 100)
Ds  =  cosmo.comoving_distance(zs)  ## units don't matter as h drops out of X * d / dX.
Ds *= h                             ## [Mpc / h]

ps  = derivative(poly_approx, Ds.value, dx=1e0)
ps *= Ds.value

pl.plot(Ds, ps, label= str(order) + r'th order polynomial')

pl.xlabel(r'$\chi$')
pl.ylabel(r'$\alpha$')

pl.xlim( 0.0, 600.0)
pl.ylim(-1.0,   2.1)

pl.legend(loc=3)

pl.savefig("BGS_alpha.pdf")
