import  matplotlib;                   matplotlib.use('PDF')

import  os 
import  math
import  time
import  warnings
import  scipy.sparse          as      sparse
import  numpy                 as      np
import  healpy                as      hp 
import  pylab                 as      pl
import  matplotlib.pyplot     as      plt

from    scipy.special         import  lpmv, sph_harm
from    scipy.sparse          import  csr_matrix, bsr_matrix, coo_matrix
from    scipy.misc            import  factorial
from    scipy.misc            import  factorial2                               as _factorial2 
from    wigtools              import  print_table
from    sympy.physics.wigner  import  wigner_3j, wigner_9j, clebsch_gordan

plt.style.use('ggplot')

warnings.filterwarnings("ignore")


start_time      = time.time()

## Goal I:  To evaluate all of Y[ind_lm][ind_pix] for ls < lmax and a healpy pixelisation defined by nside; 
lmax            = 8                                          ## lmax set by allowed p triplets considered and number of desi alms included.
nside           = 16
prec            = 1e-6                                       ## Set precision for sparsity cut. 

nalms           = hp.Alm.getsize(lmax, mmax=None)            ## Number of unique alms given lmax                                                            
ialms           = np.arange(nalms)                           ## indices to address possible alms                                                          

## Given indices to address alms, return ell and m for each of those indices.                                                                              
(ells, ms)      = hp.sphtfunc.Alm.getlm(lmax, i=ialms)       ## py:  index = l**2 + l + m + 1  or  cxx: m * (2 * lmax + 1 - m) / (2 + l)

## Ylm is non-zero on the plane (theta = pi / 2.) only if (l + m) is even.
ialms_lme       = np.where((ells + ms) % 2 == 0)[0]    
nilmes          = len(ialms_lme)
ilmes           = np.arange(nilmes)

## Healpy pixelisation of the sphere; pixel centres are on lines of constant co-latitude, have equal area and are hierarchical (base 4).                     
## Pixels in a co-latitude ring are equidistant in longitude.  Base resolution is 12 pixels.                                                                 
npix            = hp.nside2npix(nside)  ## 12 * nside**2                                                                                                    
nrings          = 4 * nside - 1         ## Number of rings of cnst. co-latitude.                                                                             
nequa           = 4 * nside
ipix            = np.arange(npix) 

## resolution as sqrt(area); pix are not squares.                                                                                                            
rt_area         = hp.pixelfunc.nside2resol(nside, arcmin=True) / 60.             ## degrees                                                                  

def print_intro():
    print("\n\nWelcome to %s"                                                                                       % os.path.basename(__file__))
    print("\nHealpy config: (nside, npix, nrings, nequa, resolution as sqrt_area [deg]) = (%d, %d, %d, %d, %.4lf)"  % (nside, npix, nrings, nequa, rt_area))
    print("Further assumed l_max of %d and a sparsity threshold of %.1le"                                           % (lmax, prec))
    print("\nY shape is (npix x nalms) = (%d, %d)"                                                                  % (npix, nalms))
    print("\nKron product: Y x Y shape is (npix x npix, nalms x nalms) = (%d, %d):"                                 % (npix * npix, nalms * nalms))

def get_phis_cthetas(planar = True, lonlat = False, ctheta=True):
    (phis, thetas)      = hp.pixelfunc.pix2ang(nside, ipix, nest=False, lonlat=True)          ## phi and theta (co-latitude) in radians.                     
    cpix                = ipix
    
    if planar is True:
        ## Using isotropy, evaluate integral in the plane with ppix; npix -> len(ipix) and set up cos(thetas) etc.                                         
        ## Use cpix as the 'calculation' pix, i.e. either ipix or ppix if limited to the plane. 

        phis            = phis[thetas == 0.0]                                                 ## phis in the plane, theta = pi / 2.
        thetas          = np.zeros_like(phis)
 
        ppix            = hp.pixelfunc.ang2pix(nside, phis, thetas, nest=False, lonlat=True)  ## ipix in the plane theta = pi / 2.
        cpix            = ppix

        print("\nY shape is (npix x nalms) = (%d, %d) in the plane."                        % (len(ppix), nalms))
        print("\nKron product: Y x Y shape is (npix x nppix, nalms x nalms) = (%d, %d):"    % (npix * len(ppix), nalms * nalms))
        
    if lonlat is False:
        phis           *= np.pi / 180.0                                                       ## radians.  
        thetas         *= np.pi / 180.0

        thetas          = np.pi / 2. - thetas

        if ctheta is True:
            thetas      = np.cos(thetas)

    return  cpix, phis, thetas

def make_sparse(matrix, prec=prec, array = False):    
    ## Zero elements below a given threshold in precision in order to create a sparse matrix.                                                        
    matrix[(np.abs(matrix.real) < prec) & (np.abs(matrix.imag) < prec)] = 0.0 + 0.0j

    sind    = np.where(matrix != 0.0 + 0.0j)
    
    print("Sparsity factor: %.6lf per cent" % (100.0 * len(sind[0]) / matrix.shape[0] / matrix.shape[1]))

    if array is True:
        matrix  = csr_matrix((matrix[sind], (sind[0], sind[1])), shape = matrix.shape).toarray()

    else:
        matrix  = csr_matrix((matrix[sind], (sind[0], sind[1])), shape = matrix.shape)

    return  matrix

def kron_prod(a, b, is_sparse=False, array=True):
    if is_sparse    is False:
        return np.kron(a, b)

    elif is_sparse  is True:
        if array    is True:
            return sparse.kron(a, b).toarray()

        else:
            return sparse.kron(a, b)
    else:
        raise ValueError("Kron prod. failed for supplied arguments.")

def get_ps(fname="txt/allowed_ps.txt", max=-1):
    ## Get allowed triplets for vec p of Tripolar harmonic; stripped from output file of wigtools.py                                                       
                                                                                                          
    ## Assumed (p1, p2, P) are the last three columns.
    allowed_ps    = np.loadtxt(fname)
    ps            = allowed_ps[:, 3:].astype(int)  ## Get the triplets as ints                                                                              
                                                                    
    ps            = ps[:max, :]                    

    print("\n\nAllowed triplets:\n")

    for i, x in enumerate(ps):
        print i, x

    return ps

def get_desialms(fname="txt/desi_alms.txt", maxrow=15):
    ## Get DESI alms from output of alm.py                                                                                                                   
    text                = np.loadtxt(fname, delimiter='\t')

    desi_alms           = [[x[0], x[2], x[4] + x[5] * 1j, x[6], x[8]] for x in text]

    desi_alms           = np.array(desi_alms, dtype=complex)

    ## Order by |a_lm|, largest first.                                                                                                                     
    desi_alms           = desi_alms[desi_alms[:,3].argsort()]
    desi_alms           = desi_alms[::-1]                            ## Largest |a_lm| first

    desi_alms           = desi_alms[:maxrow, :]

    print("\n\nDESI alms:\n")

    for i, x in enumerate(desi_alms):
        l, m, alm, alm2, Cl, ilm = strip_alm(x)

        print "%d:  %d \t %d \t %.4lf + %.4lfi \t %.4lf \t %.4lf \t %d" % (i, l, m, alm.real, alm.imag, alm2, Cl, ilm)

    return desi_alms

def strip_alm(x):
    l     =   np.int(x[0])
    m     =   np.int(x[1])

    alm   =          x[2]        ## complex

    alm2  = np.float(x[3]) **2.  ## |alm|^2
    Cl    = np.float(x[4])

    ilm   = hp.sphtfunc.Alm.getidx(lmax, l,  np.abs(m))                                      ## Healpy index corresponding to (l,m) in big alms.             
                                                                                             ## Defined for positive m only.
    return l, m, alm, alm2, Cl, ilm

def solidangle_grid(pix):
    ## Double integral over solid angle of s and s1 becomes a sum_ij over 2D matrices y[npix][npix] for given ell indexing.                         
    ## Each healpy pix corresponds to a unit vector.  Use this to populate dOmega_s and dOmega_s1 spaces.                                             
    ## i.e columns 0 to (ipix - 1) label unit norm vectors in direction specified by ipix.
    hat_vec  = np.array(hp.pixelfunc.pix2vec(nside, pix, nest=False))                        ## [3, npix]
    
    return  hat_vec

def map_cpix(cpix, indices):
    ## map cpix, e.g. [333, 334, 335] to a [0, 1, 2] array on which ylm is defined.
    return indices - cpix[0]

## Prep.
factorial2   = np.vectorize(_factorial2)

def plane_ylmprefactor(ELLs, Ms):
    ## V158(2).
    exact     = False

    interim   = 1.0
    interim  *= factorial2(ELLs + Ms - 1,  exact=exact) * factorial2(ELLs - Ms - 1, exact=exact)
    interim  /= factorial2(ELLs + Ms,      exact=exact) * factorial2(ELLs - Ms    , exact=exact)

    interim  *= (2. * ELLs + 1)
    interim  /= (4. * np.pi)

    return  np.sqrt(interim)

def ylm(phis, cthetas, planar = True):
    if planar is True:
        iLMs, PHIs                   = np.meshgrid(ilmes, phis)

        (ELLs, Ms)                   = hp.sphtfunc.Alm.getlm(lmax, i=ialms_lme)

        EXPONENT                     = (ELLs + Ms) / 2

        ## L + M > 0; (-1) ** [(ELLs + Ms) / 2]
        EXPONENT[EXPONENT % 2 == 1]  =            -1.0
        EXPONENT[EXPONENT      > 0]  =             1.0

        result                       = EXPONENT * np.exp(1j * Ms * PHIs)       
        result                      *= plane_ylmprefactor(ELLs, Ms)

        return  result

    else:
        iLMs, PHIs     = np.meshgrid(ialms,    phis)            ## Each grid point is associated to a (L,M), cos(theta) and phi.
        iLMs, CTHETAs  = np.meshgrid(ialms, cthetas)            ## Form a 2D [N_lm * Npix] lattice on which all Y are defined: 

        (ELLs, Ms)     = hp.sphtfunc.Alm.getlm(lmax, i=iLMs)    ## Convert lm index to ell, m across the grid.
    
        prefactor      = 2. * ELLs + 1.                         ## Calculate prefator of each Ylm spherical harmonic.

        prefactor     *= factorial(ELLs - Ms)
        prefactor     /= factorial(ELLs + Ms)        

        prefactor     /= 4. * np.pi

        prefactor      = np.sqrt(prefactor)

        aLs            = np.zeros_like(CTHETAs)                 ## Associated Legendre polynomials for each (L,M) and (theta, phi)
                                                                ## More efficient would be to broadcast across phi, but this is a quick calc.
        for i in iLMs[0, :]:                                    ## Get (L, M) for the first grid point. 
            (l, m)     = hp.sphtfunc.Alm.getlm(lmax, i=i)

            aLs[:, i]  = lpmv(m, l, cthetas)                    ## Calculate associated Legendre for 1D cos(theta) and each, l, m.  
                                                                ## Calculates with cos(theta) for each of Npix; could limit to co-latitude rings. 
        phases  = np.exp(1j * Ms * PHIs)

        return  prefactor * phases * aLs                        ## 2D array indexed by i_alm as defined and ipix, as defined by healpy.
    
def ylm_stwo(s, sone, hat_vec, cpix, ylm_onsphere, planar = True):
    ## Generate Ylms for (l, m), where m is positive, on s2 pixels; such that ylm_onsphere and YLM_STWO are lookup tables.
    vec_s        =    s * hat_vec
    vec_sone     = sone * hat_vec
    
    ## CHECKPIX  =     cpix[np.newaxis, :]    +  cpix[:, np.newaxis]*1j                               ## print CHECKPIX
    
    XDIFF        = vec_sone[np.newaxis, 0, :] - vec_s[0, :, np.newaxis]                               ## [npix, npix]
    YDIFF        = vec_sone[np.newaxis, 1, :] - vec_s[1, :, np.newaxis]

    if planar is False:
        ZDIFF    = vec_sone[np.newaxis, 2, :] - vec_s[2, :, np.newaxis]
        S2NORM   = np.sqrt(XDIFF**2. + YDIFF**2. + ZDIFF**2.)
        ZDIFF   /= S2NORM

        cnlms    = nalms
        cilms    = ialms

    else:
        ZDIFF    = np.zeros_like(XDIFF)
        S2NORM   = np.sqrt(XDIFF**2. + YDIFF**2.)

        cnlms    = nilmes
        cilms    =  ilmes

    XDIFF       /= S2NORM
    YDIFF       /= S2NORM
     
    S2NORM[S2NORM > 600.] = 0.0                                                                       ## Binary mask on s_2 being in -survey.       
    S2NORM[S2NORM >  0.0] = 1.0

    PIX_STWO     = np.zeros_like(XDIFF,    dtype=np.int32)                                            ## Healpy pixel at which \hat s_2 points.              
    YLM_STWO     = np.zeros((len(cpix), len(cpix),  cnlms), dtype=np.complex128)

    for i, X in enumerate(XDIFF):
        PIX_STWO[i, :]        = hp.pixelfunc.vec2pix(nside, X, YDIFF[i, :], ZDIFF[i, :], nest=False)  ## [npix, npix]; Looks completely ordered?
        PIX_STWO[i, :]        = map_cpix(cpix, PIX_STWO[i, :])                                        ## map cpix, e.g. [333, 334, 446] to [1, 4, 7]

        for j in cilms:
          YLM_STWO[i, :, j]   = ylm_onsphere['nostar'][PIX_STWO[i, :], j]
    
    ## print("Calculated Y_lms over s2 sphere in %ss" % (time.time() - start_time))
    
    return  YLM_STWO, S2NORM, vec_sone

def wrap_negm(m, sphericalharmonic, conjugate = False):
    ## Could slice by ilm before conjugate.
    if conjugate is False:
        if m > 0:
            return  sphericalharmonic['nostar']
        
        else:
            return  sphericalharmonic['star']   * (-1.)**np.abs(m) 

    else:
        if m > 0:
            return  sphericalharmonic['star']

        else:
            return  sphericalharmonic['nostar'] * (-1.)**np.abs(m)

def wrap_idx(lmax, l, m, planar = True):
    index  = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m)) 

    if planar is True:
        return np.int(np.where(ialms_lme == index)[0])

    else:
        return index

def Yxx(cpix, alms, YLM_STWO, S2NORM, ylm_onsphere, planar = True):        
    result     = np.zeros(len(cpix), dtype=np.float)[:, np.newaxis] * np.zeros(len(cpix), dtype=np.float)[np.newaxis, :]   ## result is real.                
    unit       =  np.ones(len(cpix), dtype=np.complex128)

    for x in alms:                 
      l, m, alm, alm2, Cl, ilm  = strip_alm(x)

      if((planar is True) & ((l + m) % 2 == 1)):
          ## Ylm is zero on the plane if (l + m) is odd.
          continue

      ilm      = wrap_idx(lmax, l, m, planar = planar)

      ## CHECK THIS S x S in broadcast; ## Dropped s dependence of alms.
      result  += alm2 * (wrap_negm(m, ylm_onsphere)[None, :, ilm] * unit[:, None] * wrap_negm(m, YLM_STWO, conjugate=True)[:, :, ilm]).astype(np.float)

    result *= S2NORM  ## Binary mask on s2 being in the survey; s1 in survey by definition.
    
    return result

def TPH(pone, ptwo, P, cpix, YLM_STWO, ylm_onsphere, planar = True, sparse=False, array=True):
    ## T_{p1, p2, P} Tripolar harmonic specified by a configuration: {\hat s, \hat s_1, \hat s_2}. 
    if planar is True:
        ## (l + m) must be even.
        step = 2

    else:
        step = 1

    mone     = np.arange(-pone, pone + 1, step)
    mtwo     = np.arange(-ptwo, ptwo + 1, step)
    M        = np.arange(-P,    P    + 1, step)

    ## print("Evaluating: ", pone, ptwo, P) 

    dtype    = np.float                     ## Rank-zero Tripolar harmonics are real for RSD terms.
    ## dtype = np.complex128 
             
    if sparse is False:
        result     = np.zeros(len(cpix), dtype=dtype)[:, np.newaxis] * np.zeros(len(cpix), dtype=dtype)[np.newaxis, :]

    else:
        if array is True:
            result = csr_matrix(result.shape, dtype=dtype).toarray()
        else:
            result = csr_matrix(result.shape, dtype=dtype)

    for m in mone:     ## Over s_1.         
        ilm              = wrap_idx(lmax, pone, m, planar = planar)

        for mpp in M:  ## Over s.
            ilmpp        = wrap_idx(lmax, P, mpp, planar = planar)
                
            ## CHECK THIS S x S in broadcast; ## Dropped s dependence of alms.
            interim      = wrap_negm(m, ylm_onsphere)[None, :, ilm] * wrap_negm(mpp, ylm_onsphere)[:, None, ilmpp]

            ## Kronecker product of Y_{pone, mone}(O_1) x Y_{P, M}(O_s) 
            ## interim   = kron_prod(ylm_onsphere[:, ilm], ylm_onsphere[:, ilmpp], sparse=sparse, array = array)  

            for mp in mtwo:  ## \hat s_2 determined by s, s_1, \hat s, \hat s_1
                ilmp     = wrap_idx(lmax, ptwo, mp, planar = planar)

                wig      = wigner_3j(pone, ptwo, P, m, mp, mpp)

                if(wig != 0):
                    wig      = wig.evalf()
                    wig      = np.float64(wig)

                    result  += (wig * interim * wrap_negm(mp, YLM_STWO)[:, :, mp]).astype(dtype)  ## [npix x npix].

    return result

def Q(Yxx, Tppp):
    prod = Yxx * Tppp
    
    return  np.sum(prod)


if __name__ == "__main__":
    print_intro()

    planar              = True

    cpix, phis, thetas  = get_phis_cthetas(planar = planar, lonlat = False, ctheta=False)

    ## 2D matrix Y[Npix, Nlm] which gives Ylm across a Healpix map of given nside to lmax.
    ## When restricted to the plane, (ell + m) must be even for Ylm to be non-zero.
    ylm_onsphere        = ylm(phis, np.cos(thetas), planar=planar)  

    ylm_onsphere_star   = np.conjugate(ylm_onsphere)
    
    ylm_onsphere        = {'nostar': ylm_onsphere, 'star': ylm_onsphere_star}
    
    print("\nCalculated required Y_lms on sphere in %ss" % (time.time() - start_time))

    ## print sph_harm(m, l, phis, thetas)
                
    ## ylm_onsphere     = make_sparse(ylm_onsphere)
    
    ## Defined a 2D sparse matrix Y[Npix, Nlm] and illustrated use of the sparse kronecker delta.
    ## kprod            = kron_prod(ylm_onsphere, ylm_onsphere, sparse=False, array=True)
    
    ## Allowed triplets of (p1, p2, P), ordered by B coefficient.
    trip_max            = 10
    ps                  = get_ps(max = trip_max)

    alm_max             = 30
    desi_alms           = get_desialms(maxrow = alm_max)
    
    hat_vec             = solidangle_grid(cpix)
    
    ## Main calculation:  needs to specify a s for Q(s) to evalued with an integral of mod(s_1).                                                         
    ds                  =    25.
    smax                =  1200.                                                ## Maximum comoving distance in the survey.

    sssss               = np.arange(010.0, smax, ds)
    sones               = np.arange(100.0, 600., ds)

    final_result        = []
    
    print("\n\nBeginning Qs calc. \n\n")

    for sind, s in enumerate(sssss):
      split_on_sone = []  ## This will be used to create a list of [s, Q0(s), Q1(s), etc ...], i.e. summed_on_sone.

      print("Percentage complete:  %03lf (in %03lf seconds) " % (100. * sind / len(sssss), time.time() - start_time))

      for sone in sones:
        YLM_STWO, S2NORM, vec_sone  = ylm_stwo(s, sone, hat_vec, cpix, ylm_onsphere, planar = planar)

        YLM_STWO_STAR               = np.conjugate(YLM_STWO)
        
        YLM_STWO                    = {'nostar': YLM_STWO, 'star': YLM_STWO_STAR}

        Y                           = Yxx(cpix, desi_alms, YLM_STWO, S2NORM, ylm_onsphere, planar = planar)  
        
        ## print("Calculated Yxx in %ss" % (time.time() - start_time))

        all_Qs  = [sone]  ## Given that we have s, and sone, this will be a list of the integrand for [Q0(s), Q1(s), etc ...]
        
        for trip in ps:            
            T       = TPH(trip[0], trip[1], trip[2], cpix, YLM_STWO, ylm_onsphere, planar = planar, sparse=False, array=True)
            
            ## print("Calculated T in %ss" % (time.time() - start_time))
            
            result  = Q(Y, T)
            
            all_Qs.append(result)

            ## print("\n\nFor s_one of %.6lf, answer is: %.6lf + %.6lf i." % (sone, result.real, result.imag))
    
        split_on_sone.append(all_Qs)
        
      
      split_on_sone         = np.array(split_on_sone)       
 
      split_on_sone[:, 1:]  = ds * split_on_sone[:, 0, None] ** 2. * split_on_sone[:, 1:] ## sone**2. * ds_1. 

      split_on_sone         = split_on_sone[:, 1:]                                        ## Drop sone in zeroth column.  

      summed_on_sone        = np.sum(split_on_sone, axis=0)                               ## Finally, sum over rows as integral.

      summed_on_sone        = np.insert(summed_on_sone, 0, s, axis=0)                     ## Make first column s. 

      final_result.append(summed_on_sone)                                                 ## list of [s, Q0(s), Q1(s), etc ...] for each s.
        
    final_result  = np.array(final_result)
    
    ## and save.
    np.savetxt('txt/Qs.txt', final_result)
    
    ## and load.
    final_result  = np.loadtxt("txt/Qs.txt")

    normalisation = final_result.T[1, 0]

    ## Print normed Qs.                                                                                                                                      
    final_result[:, 1:] /= normalisation

    np.savetxt('txt/Qs_normed.txt', final_result)

    final_result  = np.loadtxt("txt/Qs_normed.txt")

    ## and plot ... 
    for i, x in enumerate(final_result.T[1:, :]):
        pl.plot(final_result[:,0], x, label='', c='k', alpha=0.5)

    cmap = plt.cm.get_cmap('inferno', len(final_result.T[1:7, 0]))

    for i, x in enumerate(final_result.T[1:7, :]):
        pl.plot(final_result[:,0], x, label=tuple(ps[i, :]), c=cmap(i))

    pl.xlabel('s')
    pl.ylabel(r'$Q_{p_1 \ p_2 \ P} \ (s)$')

    pl.legend(ncol=2)

    pl.savefig("Qs.pdf")
    
    print("\n\nComplete (in %ss).\n\n" % (time.time() - start_time))
