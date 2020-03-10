import  itertools             as      it
import  numpy                 as      np

from    itertools             import  combinations_with_replacement
from    sympy.physics.wigner  import  wigner_3j, wigner_9j, clebsch_gordan


def print_table(table, max = None):
  if max is None:
    max = len(table[:,0])

  for i, row in enumerate(table):
    print ''.join(['% 6g  ' % x for x in row])

    if i > max:
      break

def return_Ls(ell, ell2):
    min  = np.abs(ell - ell2)
    max  = np.abs(ell + ell2)

    return np.arange(min, max + 1, 1)


if __name__ == "__main__":
  table  = []
  count  =  0

  ## Without selection: [0, 2], With selection: [0, 1, 2]
  ## base  = [0]
  ## base  = [0, 2] 
  base     = [0, 1, 2]

  print("\n\nWelcome to Wig tools. \n")

  for ellpair in it.product(base, repeat=2):
    Ls = return_Ls(ellpair[0], ellpair[1])

    for i, x in enumerate(it.repeat((ellpair[0], ellpair[1]), len(Ls))):
      ell_wig3j  = wigner_3j(x[0], x[1], Ls[i], 0, 0, 0).evalf()

      for jpair in it.product(base, repeat=2):
        Js = return_Ls(jpair[0], jpair[1])
      
        for k, y in enumerate(it.repeat((jpair[0], jpair[1]), len(Js))):
          jjj_wig3j  = wigner_3j(y[0], y[1], Js[k], 0, 0, 0).evalf()

          maxp       = max(x[0] + y[0], x[1] + y[1])

          for ppair in it.product(np.arange(maxp + 1), repeat=2):
            Ps = return_Ls(ppair[0], ppair[1])

            for m, z in enumerate(it.repeat((ppair[0], ppair[1]), len(Ps))):
              wig9j    = wigner_9j(x[0], y[0], z[0], x[1], y[1], z[1], Ls[i], Js[k], Ps[m], prec=64)

              cleb_p0  = clebsch_gordan(z[0],   x[0],  y[0], 0, 0, 0).evalf()  
              cleb_p1  = clebsch_gordan(z[1],   x[1],  y[1], 0, 0, 0).evalf() 
              cleb_P0  = clebsch_gordan(Ps[m], Ls[i], Js[k], 0, 0, 0).evalf() 

              if np.abs(ell_wig3j) > 0.0 and np.abs(jjj_wig3j) > 0.0 and np.abs(wig9j) > 0.0:
                if np.abs(cleb_p0) > 0.0 and np.abs(cleb_p1)   > 0.0 and np.abs(cleb_P0) > 0.0:
                  B    = (2. * x[0] + 1.) * (2. * y[0] + 1.) * (2. * x[1] + 1.) * (2. * y[1] + 1.) * (2. * Ls[i] + 1.) * (2. * Js[k] + 1.)
                  B   /= (4. * np.pi)**3.
                  B    = np.sqrt(B)

                  B   *= cleb_p0 * cleb_p1 * cleb_P0 
                  B   *= wig9j 

                  table.append([count, x[0], x[1], Ls[i], y[0], y[1], Js[k], z[0], z[1], Ps[m], B])

                  count += 1

  table = np.array(table)

  print("Unique ells: " + ''.join(['% 3d ' % x for x in np.unique(table[:,1])]))
  print("Unique   js: " + ''.join(['% 3d ' % x for x in np.unique(table[:,4])]))
  print("Unique   ps: " + ''.join(['% 3d ' % x for x in np.unique(table[:,7])]))

  print("")

  print("Unique   Ls: " + ''.join(['% 3d ' % x for x in np.unique(table[:,3])]))
  print("Unique   Js: " + ''.join(['% 3d ' % x for x in np.unique(table[:,6])]))
  print("Unique   Ps: " + ''.join(['% 3d ' % x for x in np.unique(table[:,9])]))

  ## Sort by B. 
  sorted = np.argsort(np.abs(table[:,-1]))
  sorted = sorted[::-1]

  table  = table[sorted]

  ## Check on even
  ## Check on triads.

  maxB   = table[:,-1].max()
  cent   = 0.01 * maxB

  print("\n\nTable:")
  print ''.join(['% 6s  ' % x for x in ['Row', 'ell', 'ell2', 'L', 'j_1', 'j_2', 'J', 'p_1', 'p_2', 'P', 'B']])

  print_table(table, 20)
  
  np.savetxt("table_Bs.txt", table, fmt="%d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %d \t %.6lf")

  ## Find unique p triples
  ptriplets    = [[table[i, 7], table[i, 8], table[i, 9]] for i, x in enumerate(table[:,0])]
  unique       = [list(x) for x in set(tuple(x) for x in ptriplets)]

  ## 1 <--> 2 symmetry for T.
  swapped = []

  for i, x in enumerate(unique):
    for j, y in enumerate(unique):
      if j > i:
        if((x[0] == y[1]) & (x[1] == y[0]) & (x[2] == y[2])):
          swapped.append(j)

  unique = np.delete(unique, (swapped), axis=0)

  print("\n\nUnique p triplets:")
  
  ptriplets_withB = []

  for x in unique:
    select  = table[(table[:,7] == x[0]) & (table[:,8] == x[1]) & (table[:,9] == x[2])]

    ## 1 <--> 2 symmetry.
    select  = np.concatenate([table[(table[:,7] == x[1]) & (table[:,8] == x[0]) & (table[:,9] == x[2])], select])

    lenB    = len(select[:,-1])
  
    maxB    = np.array(select[:,-1]).max()
    minB    = np.array(select[:,-1]).min()
    
    ptriplets_withB.append([maxB, minB, lenB, x[0], x[1], x[2]])


  ptriplets_withB  = np.array(ptriplets_withB)

  sorted           = np.argsort(np.abs(ptriplets_withB[:, 0]))
  sorted           = sorted[::-1]

  ptriplets_withB  = ptriplets_withB[sorted]

  cumulative_lenB  = np.cumsum(ptriplets_withB[2])

  maxB             = ptriplets_withB[0][0]

  print("\nMax B \t\t Min B \t\t % of max B \t # of terms with vec p \t vec p")

  for x in ptriplets_withB:
    print("%.6lf \t %.6lf \t %.6lf \t\t % 3d  \t\t (%d, %d, %d)" % (x[0], x[1], 100. * x[0] / maxB, x[2], x[3], x[4], x[5]))

  ## Save to text.
  np.savetxt("allowed_ps.txt", ptriplets_withB, fmt="%.6lf")

  ## Get all rows with B > 0.01 * B_max   
  table  = table[table[:,-1] > cent]

  print("\n\nNumber of unique terms: %d"        % count)
  print("Number of terms within a per cent: %d" % len(table[:,0]))

  print("\nNumber of unique p triplets: %d"     % len(unique))

  print("\n\nDone.\n\n")
