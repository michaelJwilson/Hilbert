import  matplotlib;        matplotlib.use('PDF')

import  matplotlib.pyplot  as plt
import  healpy             as hp
import  pylab              as pl
import  numpy              as np

from    mpl_toolkits.axes_grid1  import make_axes_locatable


plt.style.use('ggplot')

## print(plt.style.available)

randoms    = hp.fitsfunc.mrdfits("/global/homes/m/mjwilson/desi/randoms/randoms.fits", hdu=1)

ra         = randoms[0][::1]
dec        = randoms[1][::1]

## pl.plot(ra, dec)
## pl.savefig("/global/homes/m/mjwilson/desi/desi_footprint.pdf")

## print np.min(dec), np.min(ra)
## print np.max(dec), np.max(ra)

nside      = 128
npix       = hp.pixelfunc.nside2npix(nside)    

theta      = np.pi / 2. - np.deg2rad(dec)
phi        = np.deg2rad(ra)

ipix       = hp.pixelfunc.ang2pix(nside, theta, phi, nest=False, lonlat=False)

## print   hp.ang2pix(128, np.pi/2, 0.0)

imap       = np.zeros(npix)
imap[ipix] = 1.0

## Save to .txt
np.savetxt("desibgs_imap_nside_%d.txt" % nside, imap)

## gal_cut : float [degrees]
## pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
##
## returns a tuple (Cl, alm).

lmax        = 3*nside - 1          

(Cls, alms) = hp.sphtfunc.anafast(imap, lmax=lmax, iter=3, alm=True, pol=False, use_weights=False, datapath=None, gal_cut=0)

ells        = np.arange(len(Cls))

print("l_max:  %d" % lmax)

pl.semilogy(ells, Cls)

pl.xlim(0, 40)
pl.ylim(10**-5., 5.)

pl.ylabel(r"$C(\ell) \ = \ \sum_{m=-\ell}^{\ell} \ |a_{lm}|^2 \ / \ (2 \ell + 1)$")
pl.xlabel(r"$\ell$")

## pl.title(r"Spherical transform of DESI footprint")

pl.savefig("desi_Cls.pdf", bbox_inches="tight")

## a_lm ' s plot
pl.clf()

lstop   =  10                ## inclusive
num_ms  = (2 * lstop + 1) 

matrix  = np.zeros((num_ms, lstop + 1))

output = []

for l in np.arange(lstop + 1):    
    for j in np.arange(0, (2*l + 1), 1):
        m                    = -l + j  

        if(m >= 0): 
          index              = hp.sphtfunc.Alm.getidx(lmax, l,  m)
          alm                = alms[index]
         
        else: 
          index              = hp.sphtfunc.Alm.getidx(lmax, l, -m)                 ## For a real map, a_{l -m} = (-1)^m a_{lm} *
          alm                = np.conjugate(alms[index]) * (-1.) ** m
                             
        matrix[l-m][l]       = (alm.real**2. + alm.imag**2.) / Cls[l]
                                                      
        output.append([l, m, alm, np.absolute(alm), Cls[l]])

        for k in np.arange(num_ms):
            if(k >= (2 * l + 1)):
                matrix[k][l] = np.nan
   
        print l, m, index, alm, matrix[l-m][l]

output    = np.array(output, dtype=complex)     
## output = np.array(output, dtype=[('ell', np.float32), ('m', np.float32), ('alm', np.float32), ('mod_alm', np.float32), ('Cl', np.float32)])
## output = np.sort(output,  order='mod_alm')

fmt       = '%d \t %d \t %.6lf \t %.6lf \t %.6lf'

np.savetxt("desi_alms.txt", output, fmt=['%.6lf\t%.6lf']*5, delimiter='\t')


masked  = np.ma.array(matrix, mask = np.isnan(matrix))
cmap    = matplotlib.cm.inferno

cmap.set_bad('white', np.nan)

extent  = [0, lstop, -lstop, lstop]

im      = plt.matshow(masked, cmap=cmap, interpolation="nearest")

ax      = plt.gca()

divider = make_axes_locatable(ax)
cax     = divider.append_axes("right", size="5%", pad=0.05)

cbar    = plt.colorbar(im, cax=cax)

cbar.set_label(r'$|a_{\ell m}|^2 \ / \ C(\ell)$', size=12)

ax.set_xlabel(r"$\ell$")
ax.set_xticks(np.arange(0, lstop + 1,   1))
ax.set_xticklabels(np.arange(0, lstop + 1,   1))

ax.xaxis.set_ticks_position('bottom')

ax.set_ylabel(r"$m \rightarrow$")

## ax.set_yticks(np.arange(0, (2 * lstop + 1),    2))
## ax.set_yticklabels(np.arange(lstop, -lstop,   -2))

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',
    right='off',
    top='off',         # ticks along the top edge are off
    
    labeltop='off',
    labelleft='off',
    labelright='off',
    labelbottom='off') # labels along the bottom edge are off

ax.arrow(0, 0, 10, 10, width=0.005, head_width=0.005, head_length=0.01)

pl.savefig("desi_alms.pdf", bbox_inches="tight")

