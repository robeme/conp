# units are same as in metalwalls

import numpy as np
import math, sys
from scipy.special import gammaincc

np.set_printoptions(precision=2,suppress=True)

def main(argv):

  global wirefac, volfac, explflag, symflag, slabflag, wireflag, preflag, elcflag
  global Rc, nx, ny, nz, nmax, alpha, eta
  global kpoints, ksqmax, kprefac, kxvecs, kyvecs, kzvecs
  global Lx,Ly,Lz,V,Vinv,Axyinv,N,r
  global A,cos_kx, sin_kx, cos_ky, sin_ky, cos_kz, sin_kz
  
  symflag = False # symmetric electrodes?
  explflag = True # use explicit expression for slab correction (k=0)
  slabflag = False
  wireflag = False
  elcflag = False
  preflag = True # should ewald terms be pre-computed for speed?
  trapzflag = True # use trapezodial integration for kz long-range part

  volfac = 1.0
  
  r = np.loadtxt('metalwalls/benzene.xyz', skiprows=2, usecols=(1,2,3), dtype=float) 
  Lx = 6.440189199460001E+01    
  Ly = 6.971769370200001E+01
  Lz = 2.510765832095E+02
  
  # wrap coordinates into box with origin at lower left
  # (for a comparison with metalwalls positions should not be wrapped ...!)
  # this is strange since corrections for atoms in different images give weird results
  #for i,x in enumerate(r[:,0]): r[i,0] = x - np.floor(x / Lx) * Lx
  #for i,y in enumerate(r[:,1]): r[i,1] = y - np.floor(y / Ly) * Ly
  #for i,z in enumerate(r[:,2]): r[i,2] = z - np.floor(z / Lz) * Lz
  
  # process command line inputs
  for ninp,inp in enumerate(argv):
    if '-h' in inp: help()
    if '-f' in inp: volfac = float(argv[ninp+1])
    if '--help' in inp: help()
    if '--exp' in inp: explflag = False
    if '--sym' in inp: symflag = True
    if '--slab' in inp: slabflag = True
    if '--wire' in inp: wireflag = True
    if '--pre' in inp: preflag = False
    if '--elc' in inp: elcflag = True
    if '--trapz' in inp: trapzflag = False
    
  # user error handling  
  if slabflag and wireflag: sys.exit("ERROR: can't use slab and wire correction at same time!")
  #if slabflag and not explflag and volfac <= 1.0: sys.exit("ERROR: slab factor is too small (at least > 3.0)!")
  if wireflag and not explflag and volfac <= 1.0: sys.exit("ERROR: wire factor is too small (at least > 3.0)!")
  if slabflag and explflag and volfac > 1.0: sys.exit("ERROR: can't use explicit slab correction (EW2D) with volfac > 1.0!")
  
  # k-space parameters
  Rc = 24.0
  alpha = 1.20292E-01
  eta = 0.955234657
  # define nx, ny and nz BEFORE scaling box dimensions
  nx = 8
  ny = 9
  nz = 32
  
  # INFO: we should also increase ny and/or nz if we use EW3DC  
  #   CONP actually do and estimates nz from the modified z-dimension
  #   we could also use the method in metalwalls to estimate alpha
  #   eta and nx,ny,nz
  if wireflag: 
    Lx *= volfac
    Ly *= volfac
    nx = int(nx*volfac)
    ny = int(ny*volfac)
  if slabflag: 
    Lz *= volfac
    nz = int(nz*volfac)

  N = len(r)
  V = Lx*Ly*Lz
  Vinv = 1./V
  
  # set some constants
  nmax = max([nx,ny,nz])
  kprefac = 2*np.pi*np.array([1./Lx,1./Ly,1./Lz])
  ksqmax = np.dot(kprefac*np.array([nx,ny,nz]),kprefac*np.array([nx,ny,nz]))
  hsqmax = np.dot(kprefac[:2]*np.array([nx,ny]),kprefac[:2]*np.array([nx,ny]))
  kpoints = (2*nz+1)*(2*nx*ny+nx+ny)
  hpoints = (2*nx*ny+nx+ny)
  
  selfcorr = (np.sqrt(2)*eta-2.*alpha)/np.sqrt(np.pi)

  Rcsq = Rc*Rc

  MY_4PIVinv = 4.*np.pi*Vinv
  etasqr2inv = eta/np.sqrt(2)
  alphasq = alpha**2
  alphasqinv = 1.0 / alphasq
  sqrpialpha = np.sqrt(np.pi)/alpha
  sqrpialphainv = 1./(np.sqrt(np.pi)*alpha)
  preSk = 8.*np.pi*Vinv
  preWire = 2.*np.pi*Vinv
  Axyinv = 1./(Lx*Ly)
  MY_2PIAxyinv = 2.*np.pi/(Lx*Ly)
  MY_PIAxyinv = np.pi/(Lx*Ly)
  
  # allocate matrices 
  A = np.zeros([N,N]) 
  cos_kx = np.zeros([N,nx+1])
  sin_kx = np.zeros([N,nx+1])
  cos_ky = np.zeros([N,ny+1])
  sin_ky = np.zeros([N,ny+1])
  cos_kz = np.zeros([N,nz+1])
  sin_kz = np.zeros([N,nz+1])
  
  # precompute k-space coeffs
  print("  pre-computing k-space coeffs ...")
  if preflag: ewald_coeffs()

  ######################################
  ###        self correction         ###
  ######################################
  
  print("  calculating self corrections ...")
  for i in range(N):  
    A[i,i] += selfcorr
  
  ######################################
  ###     real-space contribution    ###
  ######################################
  
  print("  calculating real-space contributions ...")    
  for i in range(N):
    print('\r(%d/%d)' % (i+1,N), end='', flush=True)
    for j in range(i,N) if symflag else range(N):
      dx = r[j,0] - r[i,0]
      dy = r[j,1] - r[i,1]
      dz = r[j,2] - r[i,2]
      
      # minimum image convention for slab/wire geometry
      if slabflag:
        dx = dx - int(round(dx / Lx)) * Lx
        dy = dy - int(round(dy / Ly)) * Ly
      if wireflag: 
        dz = dz - int(round(dz / Lz)) * Lz
      
      dijsq = dx*dx+dy*dy+dz*dz
      
      if (i != j) & (dijsq < Rcsq): 
        dij = np.sqrt(dijsq)
        A[i,j] += ( math.erfc(alpha*dij) - math.erfc(etasqr2inv*dij) ) / dij 
        
  print('')
  
  ######################################
  ###   slab (or wire) correction    ###
  ######################################
  
  print("  calculating slab/wire corrections ...") 
  
  if wireflag:  
    if explflag:
      print("  - EW1D slab correction")
      # I'm not sure if I did here the correct thing. Maybe I need the EW1D method
      # b/c in EWD1M it is looped over first for Gx,Gy and then over Gz\neq0 while in
      # EWD1 it is looped over G\neq0 
      # -> EWD1M is most probably not working here...
      prefac = 1./(2.*Lz)
      for i in range(N):
        for j in range(i,N):
          dx = r[j,0]-r[i,0]
          dy = r[j,1]-r[i,1]
          if (dx==0) and (dy==0): continue
          
          rhosq = dx*dx+dy*dy
          alpharhosq = alphasq*rhosq
          pot_ij = prefac*(np.euler_gamma+gammaincc(0,alpharhosq)+np.log(alpharhosq))
          
          A[i,j] -= pot_ij
          if not symflag and (i != j): A[j,i] -= pot_ij 
      
    else:
      print("  - EW3DC wire correction")
      for i in range(N):
        yprefac = preWire*r[i,1]
        zprefac = preWire*r[i,2]
        for j in range(i,N) if symflag else range(N):
          pot_ij = zprefac*r[j,2]+yprefac*r[j,1]
          A[i,j] += pot_ij
          if not symflag and (i != j): A[j,i] += pot_ij

  if slabflag:
    if explflag:
      print("  - using EW2D slab correction (see Hu, JCTC, 2014)")
      for i in range(N):
        for j in range(i,N):
           zij = r[j,2] - r[i,2] 
           # multiplication of zij with erf(zij) is symmetric, thus no need for abs(...)
           zijsq = zij*zij
           pot_ij = 2.0*Axyinv * (sqrpialpha*np.exp(-zijsq*alphasq) + np.pi*zij*math.erf(zij*alpha)) 
           A[i,j] -= pot_ij
           if not symflag and (i != j): A[j,i] -= pot_ij 
    else:
      print("  - using EW3DC slab correction")
      for i in range(N):
        prefac = MY_4PIVinv*r[i,2]
        for j in range(i,N):
          pot_ij = prefac*r[j,2]
          A[i,j] += pot_ij
          if not symflag and (i != j): A[j,i] += pot_ij
    
  ######################################
  ###      k-space contribution      ###
  ######################################
  
  print("  calculating k-space contributions ... ")    
   
  if preflag:
    print("  - with precomputation")
    
    if trapzflag:
      print("  - with trapezodial integration")
      # get reciprocal lattice for 2DPBC (see metalwalls doc)
      # basically it is a summation over x and y BUT solving z
      # numerically (int -> sum) rather than using the erfc() 
      # 
      # the EW2D approach used here is discussed in Hu (JCTC, 2014)
      # 
      # it's worth to mention that this part is independent on the summation order
      # since results for k-points from LAMMPS or metalwalls give no difference
      
      for k in range(1,kpoints+1):         
        print('\r(%d/%d)' % (k,kpoints), end='', flush=True)
        
        # I did not understand why it is here enough to have the 2D k-points
        # although we actually need a EW3D for EW3DC (???). However, we get closer
        # to the results of the metalwalls EW2D results if we use the 2D k-points for EW3DC
        # However, if the use the 3D k-points from metalwalls we get the same result for 
        # as obtained using the lammps k-points from lammps_kpoints
        if explflag: l, m, n = compute_kmode_index_2D(k)
        else: l, m, n = compute_kmode_index_3D(k)
        #l, m, n = compute_kmode_index_2D(k)

        # kx = l * twopi / Lx
        kx = l*kprefac[0]
        # ky = m * twopi / Ly
        ky = m*kprefac[1]
        # kz = N * twopi / Lz
        kz = n*kprefac[2]
        
        ksq = kx*kx + ky*ky + kz*kz
        
        if ksq <= ksqmax:
        
          mabs = abs(m)
          sign_m = np.sign(m)
          nabs = abs(n)
          sign_n = np.sign(n)
          
          Sk_alpha = preSk * np.exp(-0.25 * alphasqinv * ksq) / ksq
          
          for i in range(N):
          
            cos_kxky = cos_kx[i,l] * cos_ky[i,mabs] - sin_kx[i,l] * sin_ky[i,mabs] * sign_m
            sin_kxky = sin_kx[i,l] * cos_ky[i,mabs] + cos_kx[i,l] * sin_ky[i,mabs] * sign_m
            cos_kxkykz_i = cos_kxky * cos_kz[i,nabs] - sin_kxky * sin_kz[i,nabs] * sign_n
            sin_kxkykz_i = sin_kxky * cos_kz[i,nabs] + cos_kxky * sin_kz[i,nabs] * sign_n
            for j in range(i,N):
            
              cos_kxky = cos_kx[j,l] * cos_ky[j,mabs] - sin_kx[j,l] * sin_ky[j,mabs] * sign_m
              sin_kxky = sin_kx[j,l] * cos_ky[j,mabs] + cos_kx[j,l] * sin_ky[j,mabs] * sign_m
              cos_kxkykz_j = cos_kxky * cos_kz[j,nabs] - sin_kxky * sin_kz[j,nabs] * sign_n
              sin_kxkykz_j = sin_kxky * cos_kz[j,nabs] + cos_kxky * sin_kz[j,nabs] * sign_n
              
              pot_ij = Sk_alpha * (cos_kxkykz_i*cos_kxkykz_j + sin_kxkykz_i*sin_kxkykz_j)
              
              A[i,j] += pot_ij
              if not symflag and (i != j): A[j,i] += pot_ij                 
      print('')
    else:
      print("  - with analytical expression from Hu (2014)")
      ih = 1
      hpoints = (2*nx+1)*(2*ny+1)/2
      for l in range(0,nx+1):
        for m in range(-ny,ny+1) if l>0 else range(1,ny+1):
          print('\r(%d/%d)' % (ih,hpoints), end='', flush=True)
          
          # kx = l * twopi / Lx
          hx = l*kprefac[0]
          # ky = m * twopi / Ly
          hy = m*kprefac[1]
          
          hsq = hx*hx + hy*hy
          h = np.sqrt(hsq)
         
          if hsq <= hsqmax:
          
            mabs = abs(m)
            sign_m = np.sign(m)
            
            Sk_pre = MY_PIAxyinv / h

            for i in range(N):
            
              cos_kxky_i = cos_kx[i,l] * cos_ky[i,mabs] - sin_kx[i,l] * sin_ky[i,mabs] * sign_m
              sin_kxky_i = sin_kx[i,l] * cos_ky[i,mabs] + cos_kx[i,l] * sin_ky[i,mabs] * sign_m
              for j in range(i,N):
                
                zij = r[j,2] - r[i,2] 
              
                cos_kxky_j = cos_kx[j,l] * cos_ky[j,mabs] - sin_kx[j,l] * sin_ky[j,mabs] * sign_m
                sin_kxky_j = sin_kx[j,l] * cos_ky[j,mabs] + cos_kx[j,l] * sin_ky[j,mabs] * sign_m
                
                Sk_erf = np.exp(-h*zij) * math.erfc(.5*h/alpha - alpha*zij) \
                       + np.exp( h*zij) * math.erfc(.5*h/alpha + alpha*zij)         
                
                pot_ij = Sk_erf * Sk_pre * (cos_kxky_i*cos_kxky_j + sin_kxky_i*sin_kxky_j)
                   
                A[i,j] += pot_ij
                if not symflag and (i != j): 
                  A[j,i] += pot_ij 
          ih += 1
      print(' ')
  else:
    print("  - w/o precomputation")
    if not explflag: print("  - with spherical summation geometry")
    for k in range(1,kpoints+1):
      print('\r(%d/%d)' % (k+1,kpoints), end='', flush=True)
      
      l, m, n = compute_kmode_index_2D(k)
      
      # kx = l * twopi / Lx
      kx = l*kprefac[0]
      # ky = m * twopi / Ly
      ky = m*kprefac[1]
      # kz = N * twopi / Lz
      kz = n*kprefac[2]
      
      ksq = kx*kx + ky*ky + kz*kz
      
      if ksq <= ksqmax:
        Sk_alpha = preSk * np.exp(-0.25 * alphasqinv * ksq) / ksq
        for i in range(N):
          cos_kxkykz_i = np.cos(np.dot([kx,ky,kz],r[i,:]))
          sin_kxkykz_i = np.sin(np.dot([kx,ky,kz],r[i,:]))
          for j in range(i,N):
            pot_ij = Sk_alpha * ( cos_kxkykz_i*np.cos(np.dot([kx,ky,kz],r[j,:])) \
                                + sin_kxkykz_i*np.sin(np.dot([kx,ky,kz],r[j,:])) )
            A[i,j] += pot_ij
            if not symflag and (i != j): A[j,i] += pot_ij             
    print('')
    
  ######################################
  ###         ELC correction         ###
  ######################################  
    
  if elcflag:
    print("  calculating ELC corrections ... ") 
    ih = 1
    hpoints = (2*nx+1)*(2*ny+1)/2
    for l in range(0,nx+1):
      for m in range(-ny,ny+1) if l>0 else range(1,ny+1):
        print('\r(%d/%d)' % (ih,hpoints), end='', flush=True)
      
        if l==0 and m==0: continue
      
        # kx = l * twopi / Lx
        hx = l*kprefac[0]
        # ky = m * twopi / Ly
        hy = m*kprefac[1]
        
        hsq = hx*hx + hy*hy
        h = np.sqrt(hsq)
        
        prefac = MY_2PIAxyinv*(1./(h*(1.-np.exp(h*Lz))))
        
        if hsq <= hsqmax:
        
          mabs = abs(m)
          sign_m = np.sign(m)

          for i in range(N):
          
            cos_kxky_i = cos_kx[i,l] * cos_ky[i,mabs] - sin_kx[i,l] * sin_ky[i,mabs] * sign_m
            sin_kxky_i = sin_kx[i,l] * cos_ky[i,mabs] + cos_kx[i,l] * sin_ky[i,mabs] * sign_m
            for j in range(i,N):
              
              zij = r[j,2] - r[i,2] 
            
              cos_kxky_j = cos_kx[j,l] * cos_ky[j,mabs] - sin_kx[j,l] * sin_ky[j,mabs] * sign_m
              sin_kxky_j = sin_kx[j,l] * cos_ky[j,mabs] + cos_kx[j,l] * sin_ky[j,mabs] * sign_m
              
              eih_dot_r = cos_kxky_i*cos_kxky_j + sin_kxky_i*sin_kxky_j
              pot_ij = prefac*np.cosh(h*zij)*eih_dot_r
                 
              A[i,j] += pot_ij
              if not symflag and (i != j): 
                A[j,i] += pot_ij 
        ih += 1       
    print('')
  if symflag: 
    # copy upper triangle to lower
    iu = np.triu_indices(N,1)
    il = (iu[1],iu[0])
    A[il]=A[iu]
  
  np.savetxt('A.mat',A)  
  print(A)
  
def ewald_coeffs():
  """
  setup cos and sin for Ewald summation 
  """
  
  for k in range(nx+1):
    for i in range(N):
      cos_kx[i,k] = np.cos(k*kprefac[0]*r[i,0])
      sin_kx[i,k] = np.sin(k*kprefac[0]*r[i,0])

  for k in range(ny+1):
    for i in range(N):
      cos_ky[i,k] = np.cos(k*kprefac[1]*r[i,1])
      sin_ky[i,k] = np.sin(k*kprefac[1]*r[i,1])

  for k in range(nz+1):
    for i in range(N):
      cos_kz[i,k] = np.cos(k*kprefac[2]*r[i,2])
      sin_kz[i,k] = np.sin(k*kprefac[2]*r[i,2])

def compute_kmode_index_2D(ik):
  """
  returns k vector for 2D PBC 
  
  rather than using individual loops, we compute the k-modes by running over kpoints
  
  the k-mode index is a (l,m,n) triplets
  
  assumes kmode start is (nx, ny, nz)
  l ranges from 0 to nx
  m ranges from -ny to +ny, except when l==0, it ranges from 1 to +ny
  n ranges from -nz to +nz
  """
  
  if (ik <= ny*(2*nz+1)):
    n = np.mod((ik - 1), (2*nz+1)) - nz
    m = np.floor_divide(ik - 1,2*nz+1) + 1
    l = 0
  else:
    n = np.mod((ik - 1), (2*nz+1)) - nz
    ik_mn = (ik - ny*(2*nz+1) - (n + nz) - 1) / (2*nz + 1)
    m = np.mod(ik_mn, (2*ny+1)) - ny
    l = np.floor_divide(ik_mn, 2*ny+1) + 1 
  return np.array([l,m,n],dtype=int)
  
def compute_kmode_index_3D(ik):
  """
  returns k vector for 3D PBC
  
  the k-mode index is a (l,m,n) triplets
  
  Assumes kmode start is (kmax_x, kmax_y, kmax_z)
  l ranges from 0 to nx
  m ranges from -ny to +ny, except when l==0, it ranges from 1 to +ny
  n ranges from -nz to +nz, except when l==0 and m==0, it ranges from 1 to +nz
 
  ik is the global mode index: ik=1 => k=0, l=0, n=1
  """
  if (ik <= nz):
     n = ik
     m = 0
     l = 0
  elif (ik <= nz + ny*(2*nz+1)):
     n = np.mod((ik - nz - 1), (2*nz+1)) - nz
     m = np.floor_divide(ik - nz - 1,2*nz+1) + 1 # integer division
     l = 0
  else:
     n = np.mod((ik - nz - 1), (2*nz+1)) - nz
     ik_mn = np.floor_divide(ik - nz - ny*(2*nz+1) - (n + nz) - 1,2*nz + 1) # integer division
     m = np.mod(ik_mn, (2*ny+1)) - ny
     l = np.floor_divide(ik_mn,2*ny+1) + 1 # integer division
  return np.array([l,m,n],dtype=int)

def compute_hmode_index(ih):
  """
  returns h vector for ELC correction rather than using individual loops, we compute 
  the h-modes by running over kpoints. compute_kmode_index_2D and 3D give the same 
  values if nz is set to zero.
  
  the h-mode index is a (l,m) duplet
  
  assumes kmode start is (nx, ny)
  l ranges from 0 to nx
  m ranges from -ny to +ny, except when l==0, it ranges from 1 to +ny
  """
  
  if (ih <= ny):
    n = 0
    m = (ih - 1) + 1
    l = 0
  else:
    n = 0
    ih_mn = ih - ny - 1
    m = np.mod(ih_mn, (2*ny+1)) - ny
    l = np.floor_divide(ih_mn, 2*ny+1) + 1 
  return np.array([l,m],dtype=int)

  return np.array([l,m,n],dtype=int)
      
def help():
  print('usage: python Aij.py')
  print('')
  print('  -h           print this message')
  print('  -f           set volume factor [default: %.1f]' % volfac)
  print('')
  print('  --exp        turn on explicit slab/wire correction [default: %r]' % explflag)
  print('  --sym        symmetric electrodes [default: %r]' % symflag)
  print('  --slab       toggle slab correction (z is non-periodic) [default: %r]' % slabflag)
  print('  --wire       toggle wire correction (x,y are non-periodic) [default: %r]' % wireflag)
  print('  --elc        toggle ELC correction [default: %r]' % elcflag)
  print('  --pre        toggle ewald precomputation [default: %r]' % preflag)
  print('  --help       print this message')
  
  sys.exit()

if __name__ == "__main__":
    main(sys.argv)
    
# snippet for looping over compute_kpoints() in k-space
#      print("  - with spherical (EW3D) summation order")
#      for k in range(kpoints):
#        print('\r(%d/%d)' % (step,kpoints), end='', flush=True)
#       
#        l = kxvecs[k]
#        m = kyvecs[k]
#        n = kzvecs[k]
#       
#        # kx = l * twopi / Lx
#        kx = l*kprefac[0]
#        # ky = m * twopi / Ly
#        ky = m*kprefac[1]
#        # kz = N * twopi / Lz
#        kz = n*kprefac[2]
#        
#        ksq = kx*kx + ky*ky + kz*kz
#            
#        if ksq <= ksqmax:
#          Sk_alpha = preSk * np.exp(-0.25 * alphasqinv * ksq) / ksq
#          for i in range(N):
#            cos_kxkykz_i = np.cos(np.dot([kx,ky,kz],r[i,:]))
#            sin_kxkykz_i = np.sin(np.dot([kx,ky,kz],r[i,:]))
#            for j in range(i,N):
#              pot_ij = Sk_alpha * ( cos_kxkykz_i*np.cos(np.dot([kx,ky,kz],r[j,:])) \
#                                  + sin_kxkykz_i*np.sin(np.dot([kx,ky,kz],r[j,:])) )
#              A[i,j] += pot_ij
#              if not symflag and (i != j): A[j,i] += pot_ij
#        step += 1                   
#      print('') 
# end snippet

# snippet for explicit loop over k-space
#    for l in range(0,nx+1):
#      for m in range(-ny,ny+1) if l > 0 else range(1,ny+1):
#        for n in range(1,nz+1) if not explflag and (l==0) and (m==0) else range(-nz,nz+1):
#          print('\r(%d/%d)' % (step,kpoints), end='', flush=True)
# end snippet 

#def lammps_kpoints():
#  "kpoints for k-space part"
#  global kpoints, kxvecs, kyvecs, kzvecs
#  
#  kpoints = 0 # reset kpoints
#  
#  kxvecs = []
#  kyvecs = []
#  kzvecs = []

#  # (k,0,0), (0,l,0), (0,0,m)

#  for m in range(1,nmax+1):
#    sqk = (m*kprefac[0]) * (m*kprefac[0]);
#    if sqk <= ksqmax:
#      kxvecs.append(m)
#      kyvecs.append(0)
#      kzvecs.append(0)
#      #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1
#    sqk = (m*kprefac[1]) * (m*kprefac[1]);
#    if sqk <= ksqmax:
#      kxvecs.append(0)
#      kyvecs.append(m)
#      kzvecs.append(0)
#      #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1
#    sqk = (m*kprefac[2]) * (m*kprefac[2]);
#    if sqk <= ksqmax:
#      kxvecs.append(0)
#      kyvecs.append(0)
#      kzvecs.append(m)
#      #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1

#  # 1 = (k,l,0), 2 = (k,-l,0)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[1]*l) * (kprefac[1]*l);
#      if sqk <= ksqmax:
#        kxvecs.append(k)
#        kyvecs.append(l)
#        kzvecs.append(0)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1
#        
#        kxvecs.append(k)
#        kyvecs.append(-l)
#        kzvecs.append(0)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (0,l,m), 2 = (0,l,-m)

#  for l in range(1,ny+1):
#    for m in range(1,nz+1):
#      sqk = (kprefac[1]*l) * (kprefac[1]*l) + (kprefac[2]*m) * (kprefac[2]*m)
#      if sqk <= ksqmax:
#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(m)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(-m)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (k,0,m), 2 = (k,0,-m)

#  for k in range(1,nx+1):
#    for m in range(1,nz+1):
#      sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[2]*m) * (kprefac[2]*m)
#      if sqk <= ksqmax:
#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(m)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(-m)
#        #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      for m in range(1,nz+1):
#        sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[1]*l) * (kprefac[1]*l) + (kprefac[2]*m) * (kprefac[2]*m);
#        if sqk <= ksqmax:
#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(m)
#          #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(m)
#          #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(-m)
#          #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(-m)
#          #ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1





































#  # get reciprocal lattice for 3DPBC (see metalwalls doc)    
#  for l in range(0,nx+1):
#    for m in range(1,ny+1) if l == 0 else range(ny,ny+1):
#      for n in range(1,nz+1) if (l == 0) and (m == 0) else range(nz,nz+1):



#      --------------------------------------------------------------      
#      
#      for u in range(-nx,nx+1):
#        for v in range(-ny,ny+1):
#          for w in range(-nz,nz+1):
#            # exclude k=0
#            if u+v+w == 0: continue
#            
#            kx = kprefac[0]*u
#            ky = kprefac[1]*v
#            kz = kprefac[2]*w
#            
#            ksq = np.dot([kx,ky,kz],[kx,ky,kz])
#            kdotr = np.dot([kx,ky,kz],rij)
#            
#            pref = 4.*np.pi*Vinv*np.exp(-ksq/(4.*alpha**2))/ksq
#          
#            A[i,j] += pref*np.cos(kdotr)
#
#      --------------------------------------------------------------  
#
#      for u in range(0,nx+1):
#        for v in range(-ny,ny+1) if u > 0 else range(1,ny+1):
#          for w in range(-nz,nz+1):
#            kx = kprefac[0]*u
#            ky = kprefac[1]*v
#            kz = kprefac[2]*w
#  
#            cos_kx = np.cos(kx*r[j,0])
#            sin_kx = np.sin(kx*r[j,0])
#            cos_ky = np.cos(ky*r[j,1])
#            sin_ky = np.sin(ky*r[j,1])
#            cos_kz = np.cos(kz*r[j,2])
#            sin_kz = np.sin(kz*r[j,2])
#            
#            # Compute cos/sin values using trigonometric rules
#            cos_kxky = cos_kx * cos_kx - sin_kx * sin_ky
#            sin_kxky = sin_kx * cos_ky + cos_kx * sin_ky
#            cos_kxkykz = cos_kxky * cos_kz - sin_kxky * sin_kz
#            sin_kxkykz = sin_kxky * cos_kz + cos_kxky * sin_kz
#            
#            Sk_cos[j] = Sk_cos[j] + cos_kxkykz
#            Sk_sin[j] = Sk_sin[j] + sin_kxkykz
#            
#    for j in range(i,N) if symflag else range(N):        
#      for u in range(0,nx+1):
#        for v in range(-ny,ny+1) if u > 0 else range(1,ny+1):
#          for w in range(-nz,nz+1):
#            kx = kprefac[0]*u
#            ky = kprefac[1]*v
#            kz = kprefac[2]*w
#            
#            ksq = np.dot([kx,ky,kz],[kx,ky,kz])            
#            Sk_alpha = 8.0*np.pi*Vinv*np.exp(-ksq/(4.*alpha**2))/ksq
#            
#            cos_kx = np.cos(kx*r[i,0])
#            sin_kx = np.sin(kx*r[i,0])
#            cos_ky = np.cos(ky*r[i,1])
#            sin_ky = np.sin(ky*r[i,1])
#            cos_kz = np.cos(kz*r[i,2])
#            sin_kz = np.sin(kz*r[i,2])
#            
#            # Compute cos/sin values using trigonometric rules
#            cos_kxky = cos_kx * cos_ky - sin_kx * sin_ky
#            sin_kxky = sin_kx * cos_ky + cos_kx * sin_ky
#            cos_kxkykz = cos_kxky * cos_kz - sin_kxky * sin_kz
#            sin_kxkykz = sin_kxky * cos_kz + cos_kxky * sin_kz
#            
#            A[i,j] += Sk_alpha * (Sk_cos[j] * cos_kxkykz + Sk_sin[j] * sin_kxkykz)
#
#      --------------------------------------------------------------  



#      ! Choose rkmax such that truncature error < ktol
#      !
#      ! since
#      ! error < 8 \sum_{nx}^{\infty} \sum_{ny}^{\infty} \int_{nz}^{infty} \frac{e^{-(|k|^2| + u^2)/(4\alpha^2)}}{|k|^2 + u^2} du
#      ! error < 8 * 4 * \pi \int_{rkmax}^{\infty} e^{-r^2/(4\alpha^2)} dr
#      ! error < 32 * \pi * \frac{\sqrt{\pi}}{2} * 2 * \alpha \erfc{rkmax/(2\alpha)}
#      !
#      ! we choose rkmax such that
#      ! exp(-|kcut|^2/(4*alpha^2)) = epsilon
#      !
#      rkmax = sqrt(-log(this%ktol)*4.0_wp*this%alpha*this%alpha)
#      !

#      this%nx = floor(rkmax*box%length(1)/(twopi))
#      this%ny = floor(rkmax*box%length(2)/(twopi))
#      this%knorm2_max = rkmax*rkmax

#      select case (num_pbc)
#      case(2)
#         rkmin = this%zscale_weight * twopi / box%length(3)
#         rkmax = this%zscale_range * sqrt(-log(this%ktol))*2.0_wp*this%alpha / rkmin
#         nz = floor(rkmax)
#      case(3)
#         nz = floor(rkmax*box%length(3)/(twopi))
#      case default
#         nz = 0
#      end select
#      this%nz = nz












#def ewald_coeffs():
#  "ewald coeffs for k-space part"
#  
#  global kprefac
#  global kpoints
#  
#  kprefac = 2.0*np.pi*np.array([1./Lx,1./Ly,1./Lz])
#  kpoints = 0

#  gsqxmx = kprefac[0]**2*nx**2
#  gsqymx = kprefac[1]**2*ny**2
#  gsqzmx = kprefac[2]**2*nz**2
#  gsqmx = max([gsqxmx,gsqymx,gsqzmx])
#  gsqmx *= 1.00001

#  # (k,0,0), (0,l,0), (0,0,m)

#  for m in range(1,nmax+1):
#    sqk = (m*kprefac[0]) * (m*kprefac[0]);
#    if sqk <= gsqmx:
#      kxvecs.append(m)
#      kyvecs.append(0)
#      kzvecs.append(0)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1
#    sqk = (m*kprefac[1]) * (m*kprefac[1]);
#    if sqk <= gsqmx:
#      kxvecs.append(0)
#      kyvecs.append(m)
#      kzvecs.append(0)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1
#    sqk = (m*kprefac[2]) * (m*kprefac[2]);
#    if sqk <= gsqmx:
#      kxvecs.append(0)
#      kyvecs.append(0)
#      kzvecs.append(m)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kpoints += 1

#  # 1 = (k,l,0), 2 = (k,-l,0)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[1]*l) * (kprefac[1]*l);
#      if sqk <= gsqmx:
#        kxvecs.append(k)
#        kyvecs.append(l)
#        kzvecs.append(0)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1
#        
#        kxvecs.append(k)
#        kyvecs.append(-l)
#        kzvecs.append(0)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (0,l,m), 2 = (0,l,-m)

#  for l in range(1,ny+1):
#    for m in range(1,nz+1):
#      sqk = (kprefac[1]*l) * (kprefac[1]*l) + (kprefac[2]*m) * (kprefac[2]*m)
#      if sqk <= gsqmx:
#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(-m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (k,0,m), 2 = (k,0,-m)

#  for k in range(1,nx+1):
#    for m in range(1,nz+1):
#      sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[2]*m) * (kprefac[2]*m)
#      if sqk <= gsqmx:
#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(-m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kpoints += 1

#  # 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      for m in range(1,nz+1):
#        sqk = (kprefac[0]*k) * (kprefac[0]*k) + (kprefac[1]*l) * (kprefac[1]*l) + (kprefac[2]*m) * (kprefac[2]*m);
#        if sqk <= gsqmx:
#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(-m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(-m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kpoints += 1




































