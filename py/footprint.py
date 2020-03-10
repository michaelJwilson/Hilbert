import  matplotlib;  matplotlib.use('PDF')

import  matplotlib.pyplot        as  plt
import  healpy                   as  hp
import  pylab                    as  pl
import  numpy                    as  np
import  ephem

from    mpl_toolkits.axes_grid1  import make_axes_locatable


plt.style.use('ggplot')

fig         = plt.figure(figsize=(10, 5))
ax          = fig.add_subplot(111, projection='mollweide', facecolor = 'gainsboro')

def plot_mwd(RA, Dec, weights=None, org=0, title='DESI', color='k', s=0.2, alpha=1.0):
    ''' 
    RA [0, 360), Dec [-90, 90] degrees are arrays of the same length.

    org is the origin of the plot, 0 or a multiple of 30 degrees in [0, 360).
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    '''
    
    if weights is None:
        weights = np.ones_like(RA)

    RA      =  RA[weights > 0.0]
    Dec     = Dec[weights > 0.0]

    x       =  np.remainder(RA + 360 - org, 360)  # shift RA values
    ind     =  x > 180
    x[ind] -=  360                                # scale conversion to [-180, 180]
    x       = -x                                  # reverse the scale: East to the left

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + org,360)

    ax.scatter(np.radians(x), np.radians(Dec), marker='.', color=color, s=s, rasterized=True, alpha=alpha)  

    ax.set_xticklabels(tick_labels)                        
    ## ax.set_title(title)
    ax.title.set_fontsize(15)
    ax.set_xlabel("right ascension")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("declination")
    ax.yaxis.label.set_fontsize(12)

    ax.grid(True)


##  http://balbuceosastropy.blogspot.co.uk/2013/09/the-mollweide-projection.html                                                                            
randoms     = hp.fitsfunc.mrdfits("/global/homes/m/mjwilson/desi/randoms/randoms.fits", hdu=1)
rand_ra     =   randoms[0][::50]
rand_dec    =   randoms[1][::50]

lognormal   = hp.fitsfunc.mrdfits("/global/homes/m/mjwilson/desi/lognormal_bgs.fits",   hdu=1)
lognorm_ra  = lognormal[0]
lognorm_dec = lognormal[1]
lognorm_zee = lognormal[2]
lognorm_bgs = lognormal[3]

print lognorm_zee.min(), lognorm_zee.max()

lognorm_bgs[lognorm_zee > 0.5] = 0.0

## plot_mwd(rand_ra,    rand_dec,    org=90, title ='DESI footprint', color='dodgerblue', alpha=0.1)
plot_mwd(lognorm_ra, lognorm_dec, weights=lognorm_bgs, org=90, title ='DESI footprint', color='firebrick',  alpha=0.1)

lon_array  = np.linspace(0., 360., 5000)
lat        = 0.0

eq_array   = np.zeros((len(lon_array), 2))

for i, lon in enumerate(lon_array):
    ga            = ephem.Galactic(np.radians(lon), np.radians(lat))
    eq            = ephem.Equatorial(ga)

    eq_array[i]   = np.degrees(eq.get())

RA  = eq_array[:,0]
Dec = eq_array[:,1]

plot_mwd(RA, Dec, 180, color='k', alpha=1.0, s=0.5)

pl.savefig("footprint.pdf")


