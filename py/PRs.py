import  matplotlib;                   matplotlib.use('PDF')

import  math

import  time
import  warnings
import  scipy.sparse          as      sparse
import  numpy                 as      np
import  healpy                as      hp
import  pylab                 as      pl
import  itertools             as      it
import  matplotlib.pyplot     as      plt

from    scipy.special         import  lpmv
from    scipy.sparse          import  csr_matrix, bsr_matrix, coo_matrix
from    scipy.misc            import  factorial
from    wigtools              import  print_table, return_Ls
from    sympy.physics.wigner  import  wigner_3j, wigner_9j, clebsch_gordan
from    Qs                    import  *

plt.style.use('ggplot')

warnings.filterwarnings("ignore")


def Wxx(configs, alms, alms_max):
    for config in configs:
        cpix        = configs[config]['pix']
        YLM_SXX     = configs[config]['YLM_SXX']
        SMASK       = configs[config]['SMASK']
        ylm_onpix   = configs[config]['YLM_onpix']
        
        result      = np.zeros(len(cpix), dtype=np.complex128)[:, np.newaxis] * np.zeros(len(cpix), dtype=np.complex128)[np.newaxis, :]
        unit        =  np.ones(len(cpix), dtype=np.complex128)
        
        for i, x in enumerate(alms[:alms_max, :]):
            l, m, alm, nalm, Cl, ilm = strip_alm(x)

            for j, y in enumerate(alms[:alms_max, :]):
                ll, mm, allmm, nallmm, Cll, illmm = strip_alm(y)

                ##  Dropped s dependence of alms.                                                                      
                term     = alm * allmm * unit[:, None] * wrap_negm(m, ilm, ylm_onpix)[:, None, ilm] * np.conjugate(wrap_negm(mm, illmm, YLM_SXX)[:,:,illmm])

                ## print("Contribution: %.6lf \t %.6lf \t %.6lf" % (alm, allmm, alm * allmm))

                result  += term

        result   *= SMASK   ## Binary mask on s2 / s3 being in the survey.                                                                                
        
        ## result = make_sparse(result, prec = prec)

        configs[config]['result'] = result                                                                                              

    configs['prod'] = kron_prod(configs['one']['result'], configs['four']['result'], is_sparse = True, array = False)

    return result

def allowed_hexlets(configs):
    base = [0, 1, 2]

    print("\n\nAllowed hexlets:\n")

    for p12pair in it.product(base, repeat=2):
        P12s = return_Ls(p12pair[0], p12pair[1])

        for P12 in P12s:
            for p34pair in it.product(base, repeat=2):
                P34s = return_Ls(p34pair[0], p34pair[1])

                for P34 in P34s:
                    for p56pair in it.product(base, repeat=2):
                        P56s = return_Ls(p56pair[0], p56pair[1])
                        
                        for P56 in P56s: 
                            HPH(p12pair[0], p12pair[1], P12, p34pair[0], p34pair[1], P34, p56pair[0], p56pair[1], P56, configs, nonzero = True) 

def HPH(p1, p2, p12, p3, p4, p34, p5, p6, p56, configs, nonzero = False):
    ## H_{p1, p2, p12, ... }:  Hexapolar Spherical Harmonics specified by a double configuration: {\hat s, \hat s_1, \hat s_2} + primed.                  

    m1      = np.arange(-p1,  p1  + 1, 1)
    m2      = np.arange(-p2,  p2  + 1, 1)
    m12     = np.arange(-p12, p12 + 1, 1)

    m3      = np.arange(-p3,  p3  + 1, 1)
    m4      = np.arange(-p4,  p4  + 1, 1)
    m34     = np.arange(-p34, p34 + 1, 1)

    m5      = np.arange(-p5,  p5  + 1, 1)
    m6      = np.arange(-p6,  p6  + 1, 1)
    m56     = np.arange(-p56, p56 + 1, 1)

    if nonzero is False:
      cpix        = configs['one']['pix']
      YLM_SXX     = configs['one']['YLM_SXX']
      SMASK       = configs['one']['SMASK']
      ylm_onpix   = configs['one']['YLM_onpix']
                
      for p1m in m1:  ## Over s_1                                                                                                                          
        ip1m    = hp.sphtfunc.Alm.getidx(lmax, p1, np.abs(p1m))  ## index for pone, mone                                                               

        for p2m in m2:  ## Over s                                                                                                                        
          ip2m  = hp.sphtfunc.Alm.getidx(lmax, p2, np.abs(p2m))  ## index for P, M                                                              

          interim    = wrap_negm(p1m, ip1m, configs['one']['YLM_onpix'])[:, None, ip1m] * wrap_negm(p2m, ip2m, configs['one']['YLM_onpix'])[None, :, ip2m]
 
          for p12m in m12:  ## \hat s_2 determined by s, s_1, \hat s, \hat s_1                                                                         
            ip12m    = hp.sphtfunc.Alm.getidx(lmax, p12, np.abs(p12m))  ## index for ptwo, mtwo.                                                     

            c12      = clebsch_gordan(p12, p12m, p1, p1m, p2, p2m)

            if(c12 != 0):
              c12      = c12.evalf()
              c12      = np.float64(c12)

              for p3m in m3:
                ip3m   = hp.sphtfunc.Alm.getidx(lmax, p3,  np.abs(p3m))

                result   = c12 * interim * wrap_negm(p3m, ip3m, configs['one']['YLM_SXX'])[:, :, p3m]
                
                for p4m in m4:
                  ip4m   = hp.sphtfunc.Alm.getidx(lmax, p4,  np.abs(p4m))
                
                  for p34m in m34:
                    ip34m  = hp.sphtfunc.Alm.getidx(lmax, p34, np.abs(p34m))

                    c34    = clebsch_gordan(p34, p34m, p3, p3m, p4, p4m)

                    if(c34 != 0):
                      c34      = c34.evalf()
                      c34      = np.float64(c34)

                      for p5m in m5:
                        ip5m    = hp.sphtfunc.Alm.getidx(lmax, p5,  np.abs(p5m))

                        interim = wrap_negm(p1m, ip1m, configs['one']['YLM_onpix'])[:, None, ip1m] * \
                                  wrap_negm(p2m, ip2m, configs['one']['YLM_onpix'])[None, :, ip2m]

                        for p6m in m6:
                            ip6m   = hp.sphtfunc.Alm.getidx(lmax, p6,  np.abs(p6m))

                            for p56m in m56:
                                ip56m  = hp.sphtfunc.Alm.getidx(lmax, p56,  np.abs(p56m))

                                c56    = clebsch_gordan(p56, p56m, p5, p5m, p6, p6m)

                                if(c56 != 0):
                                    c56    = c56.evalf()                                    
                                    c56    = np.float64(c56)
                  
                                    c1234  = clebsch_gordan(p56, -p56m, p12, p12m, p34, p34m) 

                                    ## From C^00; pg. 248 of Varshalovich.
                                    c1234 *= (-1) ** (p56 + p56m) / np.sqrt(2. * p56 + 1.) 
                  
                                    if(c1234  != 0):
                                        c1234  = c1234.evalf()
                                        c1234  = np.float64(c1234)

                                        if nonzero is True:
                                            print p1, p2, p12, p3, p4, p34, p5, p6, p56

                                            return
                                        
                                        else:
                                            term  = alm * allmm * unit[:, None] * wrap_negm(m, ilm, ylm_onpix)[:, None, ilm] * np.conjugate(wrap_negm(mm, illmm, YLM_SXX)[:,:,illmm])
                                            
                                            print  c12 * c34 * c56 * c1234

if __name__ == "__main__":
    print_intro()
    
    ##  Need both planar and sphere defined versions.
    ipix,  phis,  cthetas  = get_phis_cthetas(planar = False)
    ppix, pphis, pcthetas  = get_phis_cthetas(planar =  True)
    
    ## Need to fix up ylm_onsphere in Qs.py
    ylm_onsphere           = ylm( phis,  cthetas, ialms)
    ylm_onplane            = ylm(pphis, pcthetas, ialms)

    ## Allowed tr iplets of (p1, p2, P), ordered by B coefficient.                                                                                          
    ps                     = get_ps()

    alm_max                = 10

    desi_alms              = get_desialms()
    
    hat_vec                = solidangle_grid(ipix)
    phat_vec               = solidangle_grid(ppix)
    
    ## Main calculation:  needs to specify a s for Q(s) to evalued with an integral of mod(s_1).                                                             
    ds                     = 25.                             ## Scale on which geometry varies. 
    dsone                  = 50.                             ## Scale on which radial selection varies. 

    smin                   =  50.                            ## Where do you trust linear theory.  
    smax                   = 250.                            ## Maximum separation in the survey. 

    sssss                  = np.arange(smin, smax, ds)       ## Limited by maximum separation in the survey. 
    ttttt                  = np.arange(smin, smax, ds)
    
    sones                  = np.arange(dsone, 600., dsone)   ## Limited by maximum radial distance from the observer.
    sfours                 = np.arange(dsone, 600., dsone)

    final_result           = []

    start_time             = time.time()
    
    iterations             = len(sssss) * len(ttttt) * len(sones) * len(sfours)

    YLM_STWO,   S2NORM, vec_sone   = ylm_stwo(ds, 250., phat_vec, ppix, ylm_onplane)   ## Assumed planar.
    YLM_STHREE, S3NORM, vec_sfour  = ylm_stwo(ds, 250.,  hat_vec, ipix, ylm_onsphere)  ## Arbitrary orientation.

    configs    = { 'one': {'pix': ppix, 'YLM_SXX': YLM_STWO,   'SMASK': S2NORM, 'YLM_onpix': ylm_onplane}, \
                  'four': {'pix': ipix, 'YLM_SXX': YLM_STHREE, 'SMASK': S3NORM, 'YLM_onpix': ylm_onsphere}}

    ## Wxx(configs, desi_alms, alm_max)

    allowed_hexlets(configs)

    ## HPH(0, 0, 0, 0, 0, 0, 0, 0, 0, configs)

    calc_time                      = time.time() - start_time

    print("\n\nTime for calc: %s [secs], iterations: %d, total calc: %.6lf [hrs]" % (calc_time, iterations, iterations * calc_time / 60. / 60.))  
    
    '''
    stride = 10
    
    for s in sssss[::stride]:
      split_on_sone = []  ## This will be used to create a list of [s, Q0(s), Q1(s), etc ...], i.e. summed_on_sone.                                         

      for t in ttttt[::stride]:
          split_on_sfour = []

          print(s, t) 
          
          for sone in sones[::stride]:
              YLM_STWO,   S2NORM, vec_sone  = ylm_stwo(s,  sone,   phat_vec, ppix, ylm_onplane)   ## Assumed planar.

          for sfour in sfours[::stride]:
              YLM_STHREE, S3NORM, vec_sfour = ylm_stwo(t,  sfour,   hat_vec, ipix, ylm_onsphere)  ## Arbitrary orientation.
    '''

    print("\n\nDone.\n\n")

