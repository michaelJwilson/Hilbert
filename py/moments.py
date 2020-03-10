import  matplotlib;                   matplotlib.use('PDF')
 
import  numpy                 as      np
import  pylab                 as      pl
import  itertools             as      it
import  matplotlib.pyplot     as      plt
import  itertools             as      it 

from    mcfit.cosmology       import  *
from    mcfit                 import  kernels, mcfit, transforms
from    nbodykit.lab          import  cosmology                        
from    sympy.physics.wigner  import  wigner_3j, wigner_9j, clebsch_gordan
from    wigtools              import  return_Ls, print_table
from    Qs                    import  get_ps
from    scipy.interpolate     import  interp1d

plt.style.use('ggplot')


redshift   = 0.2
cosmo      = cosmology.Planck15

transfers  = ['CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu']

Plin       = cosmology.LinearPower(cosmo, redshift, transfer = transfers[1])

## Note:  Plin.sigma8
##        Plin.velocity_dispersion(kmin = 1e-05, kmax = 10.0)

k          = np.logspace(-3., 3., num = 500, endpoint=False)

n          = 0
L          = 0

Pk         = Plin(k)

tPk        = Pk / k ** n 
tPk       /= (2. * np.pi ** 2.)
tPk       /= np.sqrt(2. / np.pi)

ns         = [0, 1, 2, 3, 4]
Ls         = [0, 1, 2, 3, 4]

pairs      = list(it.product(ns, Ls))

moments     = {}
interp_xinl = {}

for pair in pairs[:5]:
    (n, L)    = pair

    s, xi_nL  = transforms.SphericalBessel(k, nu = L, q = 1.5, N = None, lowring = True)(tPk)

    pl.plot(s, s * s * xi_nL, label=r'$(%d, %d)$' % (n, L))

    moments[pair]     = xi_nL

    interp_xinl[pair] = interp1d(s, xi_nL, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=True)

for l in [0, 2, 4]:
    s, xi     = P2xi(k, l=l, q=1.5, N=None, lowring=True)(Pk)

    pl.plot(s, (0.0 + 1.j) ** l * s * s * xi, '--', label=r'$%d$' % l)

pl.xlim(+00.0, 200.0)
pl.ylim(-10.0, 100.0)

plt.xlabel(r"$s$ [$h^{-1} \ \mathrm{Mpc}$]")
plt.ylabel(r"$\xi^{n}_L(s)$")

pl.legend(ncol = 4)

pl.savefig('xi_nL.pdf', bbox_inches="tight")

beta     = 0.5
b1       = 1.0

def szalay_beta(config):
    ## Assumes alpha terms are unity. 

    if config   is [0, 0, 0]:
        return (1. + beta / 3.)**2.

    elif config is [0, 2, 2]:
        return (4. * beta **2. / 9.)

    elif config is [0, 0, 2] or [0, 2, 0]:
        return (2. * beta / 3.) * (1. + beta / 3.)
        
    elif config is [1, 1, 0] or [1, 0, 1]:
        return beta * (1.0 + beta / 3.)

    elif config is [1, 1, 2] or [1, 2, 1]:
        return 2. * beta * beta / 3.

    elif config is [2, 1, 1]:
        return - beta * beta

    else:
        raise ValueError("Term not present in Szalay xi.")

## Configuration space moments plot; no alpha terms.
pl.clf()
'''
base     = [0, 2]
trips    = []

for (ell, ellp) in it.product(base, repeat=2):
    Js   = return_Ls(ell, ellp)

    for i, j in enumerate(it.repeat((ell, ellp), len(Js))):
        trips.append([j[0], j[1], Js[i]])

trips = np.array(trips)

def XI(trip): 
    result  = wigner_3j(trip[0], trip[1], trip[2], 0, 0, 0).evalf() * szalay_beta([0, trip[0], trip[1]]) * moments[(0, trip[2])]

    result /= np.sqrt((2. * trip[0] + 1.) * (2. * trip[1] + 1.))

    result *= (4. * np.pi)**(3./2.)

    result *= (-1) ** trip[1]
    
    return  result

def Fk(trip):
    result  = wigner_3j(trip[0], trip[1], trip[2], 0, 0, 0).evalf() * szalay_beta([0, trip[0], trip[1]]) * Pk / (4. * np.pi)

    result /= np.sqrt((2. * trip[0] + 1.) * (2. * trip[1] + 1.))

    result *= (4. * np.pi)**(3./2.)

    result *= (-1) ** trip[1]

    return  result

cmap = plt.cm.get_cmap('inferno', len(trips[:,0]))

for i, trip in enumerate(trips):
    pl.plot(s, s * s * XI(trip), c = cmap(i), label=str(tuple(trip)))

pl.xlim(+000.0, 400.0)
pl.ylim(-100.0, 550.0)

plt.xlabel(r"$s$ [$h^{-1} \ \mathrm{Mpc}$]")
plt.ylabel(r"$s^2 \ \Xi_{j_1 \ j_2 \ J} \ (s)$")

pl.legend(ncol = 3)

pl.savefig("XI.pdf", bbox_inches="tight")
'''
'''
def Kaiser(trip):
    if   trip[2] == 0:
         result  = 1. + (2. / 3.) * beta + beta * beta / 5.

    elif trip[2] == 2:
         result  = (4. / 3.) * beta + (4. / 7.) * beta * beta

    elif trip[2] == 4:
         result  = (8. / 35.) * beta * beta

    else:
         result  = 0.0

    result  *= Pk 
    result  *= 2. / (2. * trip[2] + 1.)

    return  wigner_3j(trip[0], trip[1], trip[2], 0, 0, 0).evalf() * result
'''
'''
## Spectral space moments plot.
pl.clf()

for i, trip in enumerate(trips):
    pl.loglog(k,      Fk(trip),       c = cmap(i), label=str(tuple(trip)))
    pl.loglog(k,  Kaiser(trip), '--', c = cmap(i), label='')

pl.xlim(0.001,   1e0)
pl.ylim(10.00,   1e5)

plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$F_{j_1 \ j_2 \ J} \ (k)$")

pl.legend(ncol = 4)

pl.savefig("Fk.pdf", bbox_inches="tight")
'''

## Masked spectral moments plot. 
ps     = get_ps(fname="allowed_ps.txt") ## Allowed triplets of (p1, p2, P), ordered by B coefficient.

Qs     = np.loadtxt("Qs_normed.txt")    ## s followed by Qs ordered by triplets. 
table  = np.loadtxt("table_Bs.txt")     

Fs     = {}

## Initialise
for t in table:
    jtrip     = (t[4], t[5], t[6])    

    Fs[jtrip] = 0.0

for t in table:
    n          = 0

    L          = t[3]
    J          = t[6]
    P          = t[9]

    B          = t[-1]

    ltrip      = (t[1], t[2], t[3])
    jtrip      = (t[4], t[5], t[6])
    ptrip      = (t[7], t[8], t[9])

    index      = np.where((ps[:, 2] == ptrip[2]) & (((ps[:, 0] == ptrip[0]) & (ps[:, 1] == ptrip[1])) | ((ps[:, 0] == ptrip[1]) & (ps[:, 1] == ptrip[0]))))

    try:
        result     = Qs[:, 1 + index[0]] * wigner_3j(ltrip[0], ltrip[1], ltrip[2], 0, 0, 0).evalf()   ## Q given p trip.

        result    *= B

        result    *= interp_xinl[(n, L)](Qs[:, 0])[:, None]                 ## Interpolate xi^n_L from FFTlog s to s on which Qs are defined. 
        
        result    *= wigner_3j(ltrip[0], ltrip[1], ltrip[2], 0, 0, 0).evalf() * (2. * L + 1.)
    
        result    *= (0.0 + 1.0j)**L
        result    /= (0.0 + 1.0j)**n

        result    *= -1**ltrip[1]
        
        result    *= szalay_beta([n, ltrip[0], ltrip[1]])

        ## print  ptrip, ps[index], result[0]

        Fs[jtrip] += result

    except:
        ## Missing e.g. (4.0, 4.0, 6.0) from ps, Qs, Bs. 
        result     = np.zeros_like(Fs[jtrip])

        print "ERROR:", ptrip, ps[index], result[0]

        Fs[jtrip] += result

pl.clf()

for x in Fs.keys()[:1]:
    print("\n\nCalculating for J of %d.\n" % x[2])
    
    J          = x[2]
 
    Fs[x]     /= np.sqrt(2. / np.pi)

    Fss        = np.zeros_like(Qs[:,0])
    ss         = np.zeros_like(Qs[:,0])
    
    for i, y in enumerate(Qs[:,0]):
        ss[i]  = y
        Fss[i] = Fs[x][i]

        print ss[i], Fss[i]

    interp_Fs  = interp1d(ss, Fss, kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=True)

    logs_Fs    = interp_Fs(s)

    k, Fk      = transforms.SphericalBessel(s, nu = J, q = 0.0, N = None, lowring = True)(logs_Fs)

    print Fk

    pl.loglog(k, np.abs(Fk), label = tuple(x))
    

pl.xlim(0.001,   1e0)                                                                                                                                      
## pl.ylim(10.00,   1e5)                                                                                                                                   
 
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")                                                                                                               
plt.ylabel(r"$F_{j_1 \ j_2 \ J} \ (k)$")

pl.legend()

pl.savefig("masked_Fk.pdf", bbox_inches="tight")

print("\n\nDone.\n\n")
