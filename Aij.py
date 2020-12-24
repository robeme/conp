import numpy as np
import math, itertools

symflag = False

slabfac = 3.0
wirefac = 1.0

Rc = 12.0

# points need to be sorted to which
# electrode they belong and how 
# they appear in the data file 
# (compare points.data)
r = np.array([[0.,0.,15.],
              [0.,0.,10.],
              [0.,0.,-15.],
              [0.,0.,-10.]]) # 1 1 2 2
N = len(r)
Lx = 31.
Ly = 31.*wirefac
Lz = 31.*slabfac
V = Lx*Ly*Lz
Vinv = 1./V
Axyinv = 1./(Lx*Ly)

# k-space
nx = 4
ny = 4 
nz = 9
nmax = max([nx,ny,nz])

kprefac = 2*np.pi*np.array([1./Lx,1./Ly,1./Lz])
ksqmax = np.dot(kprefac*np.array([nx,ny,nz]),kprefac*np.array([nx,ny,nz]))
kpoints = (2*nz+1)*(2*nx*ny+nx+ny)

cos_kx = np.zeros([N,nx+1])
sin_kx = np.zeros([N,nx+1])
cos_ky = np.zeros([N,ny+1])
sin_ky = np.zeros([N,ny+1])
cos_kz = np.zeros([N,nz+1])
sin_kz = np.zeros([N,nz+1])

alpha = 0.1780932
eta = 1.979

# set some constants
etasqr2 = eta/np.sqrt(2)
alphasq = alpha**2
alphasqinv = 1.0 / alphasq
sqrpialpha = np.sqrt(np.pi)/alpha
preSk = 8.0 * np.pi * Vinv

selfcorr = (np.sqrt(2)*eta-2.*alpha)/np.sqrt(np.pi)

def main():
  
  # precompute k-space coeffs
  ewald_coeffs()

  # allocate Aij
  A = np.zeros([N,N]) 

  for i in range(N):
    
    ######################################
    ###        self correction         ###
    ######################################
    
    A[i,i] += selfcorr
    
    for j in range(i,N) if symflag else range(N):
      # slab or wire pbc distances
      dx = np.mod(r[i,0]-r[j,0], Lx*.5)
      dy = r[i,1]-r[j,1] if wirefac > 1. else np.mod(r[i,1]-r[j,1], Ly*.5)
      dz = r[i,2]-r[j,2]
      rij = np.array([dx,dy,dz])
      dij = np.linalg.norm(rij)
      
    ######################################
    ###     real-space contribution    ###
    ######################################
    
      if (i != j) & (dij < Rc): A[i,j] += ( math.erfc(alpha*dij) - math.erfc(etasqr2*dij) ) / dij 

      # slab (or wire) correction
      if wirefac > 1.: A[i,j] += 2.*np.pi*Vinv*(r[i,2]*r[j,2]+r[i,1]*r[j,1])
      else: A[i,j] += 4.*np.pi*Vinv*r[i,2]*r[j,2]
    
    ######################################
    ###      k-space contribution      ###
    ######################################
    
  # get reciprocal lattice for 2DPBC (see metalwalls doc)
  for l in range(0,nx+1):
    for m in range(-ny,ny+1) if l > 0 else range(1,ny+1):
      for n in range(-nz,nz+1):
       
        # kx = l * twopi / Lx
        kx = l*kprefac[0]
        # ky = m * twopi / Ly
        ky = m*kprefac[1]
        # kz = N * twopi / Lz
        kz = n*kprefac[2]
        
        ksq = np.dot([kx,ky,kz],[kx,ky,kz])
        
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
            # TODO: no matter what this contribution is actually symmetric
            for j in range(i,N) if symflag else range(N):
            
              cos_kxky = cos_kx[j,l] * cos_ky[j,mabs] - sin_kx[j,l] * sin_ky[j,mabs] * sign_m
              sin_kxky = sin_kx[j,l] * cos_ky[j,mabs] + cos_kx[j,l] * sin_ky[j,mabs] * sign_m
              cos_kxkykz_j = cos_kxky * cos_kz[j,nabs] - sin_kxky * sin_kz[j,nabs] * sign_n
              sin_kxkykz_j = sin_kxky * cos_kz[j,nabs] + cos_kxky * sin_kz[j,nabs] * sign_n
            
              A[i,j] += Sk_alpha * (cos_kxkykz_i*cos_kxkykz_j + sin_kxkykz_i*sin_kxkykz_j)
              
    ######################################
    ###        k=0 contribution        ###
    ######################################
  for i in range(N):
    # TODO: no matter what this contribution is actually symmetric
    for j in range(N):
       zij = r[j,2] - r[i,2]
       zijsq = zij*zij
       A[i,j] -= 2.0*Axyinv * (sqrpialpha*np.exp(-zijsq*alphasq) + np.pi*zij*math.erf(zij*alpha))
      
  if symflag: 
    # copy upper triangle to lower
    iu = np.triu_indices(N,1)
    il = (iu[1],iu[0])
    A[il]=A[iu]
    
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

if __name__ == "__main__":
    main()


#  # get reciprocal lattice for 3DPBC (see metalwalls doc)    
#  for l in range(0,nx+1):
#    for m in range(1,ny+1) if l == 0 else range(ny,ny+1):
#      for n in range(1,nz+1) if (l == 0) and (m == 0) else range(nz,nz+1):

#def compute_kmode_index(ik):
#  """
#  compute k-mode index
#  
#  the k-mode index is a (l,m,n) triplets
#  
#  assumes kmode start is (kmax_x, kmax_y, kmax_z)
#  l ranges from 0 to kmax_x
#  m ranges from -kmax_y to +kmax_y, except when l==0, it ranges from 1 to +kmax_y
#  n ranges from -kmax_z to +kmax_z
#  """
#  
#  if (ik <= ny*(2*nz+1)):
#    n = np.mod((ik - 1), (2*nz+1)) - nz
#    m = np.floor_divide(ik - 1,2*nz+1) + 1
#    l = 0
#  else:
#    n = np.mod((ik - 1), (2*nz+1)) - nz
#    ik_mn = (ik - ny*(2*nz+1) - (n + nz) - 1) / (2*nz + 1)
#    m = np.mod(ik_mn, (2*ny+1)) - ny
#    l = np.floor_divide(ik_mn, 2*ny+1) + 1 
#  return np.array([l,m,n])

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
#      ! error < 8 \sum_{kmax_x}^{\infty} \sum_{kmax_y}^{\infty} \int_{kmax_z}^{infty} \frac{e^{-(|k|^2| + u^2)/(4\alpha^2)}}{|k|^2 + u^2} du
#      ! error < 8 * 4 * \pi \int_{rkmax}^{\infty} e^{-r^2/(4\alpha^2)} dr
#      ! error < 32 * \pi * \frac{\sqrt{\pi}}{2} * 2 * \alpha \erfc{rkmax/(2\alpha)}
#      !
#      ! we choose rkmax such that
#      ! exp(-|kcut|^2/(4*alpha^2)) = epsilon
#      !
#      rkmax = sqrt(-log(this%ktol)*4.0_wp*this%alpha*this%alpha)
#      !

#      this%kmax_x = floor(rkmax*box%length(1)/(twopi))
#      this%kmax_y = floor(rkmax*box%length(2)/(twopi))
#      this%knorm2_max = rkmax*rkmax

#      select case (num_pbc)
#      case(2)
#         rkmin = this%zscale_weight * twopi / box%length(3)
#         rkmax = this%zscale_range * sqrt(-log(this%ktol))*2.0_wp*this%alpha / rkmin
#         kmax_z = floor(rkmax)
#      case(3)
#         kmax_z = floor(rkmax*box%length(3)/(twopi))
#      case default
#         kmax_z = 0
#      end select
#      this%kmax_z = kmax_z












#def ewald_coeffs():
#  "ewald coeffs for k-space part"
#  
#  global unitk
#  global kcount
#  
#  unitk = 2.0*np.pi*np.array([1./Lx,1./Ly,1./Lz])
#  kcount = 0

#  gsqxmx = unitk[0]**2*nx**2
#  gsqymx = unitk[1]**2*ny**2
#  gsqzmx = unitk[2]**2*nz**2
#  gsqmx = max([gsqxmx,gsqymx,gsqzmx])
#  gsqmx *= 1.00001

#  # (k,0,0), (0,l,0), (0,0,m)

#  for m in range(1,nmax+1):
#    sqk = (m*unitk[0]) * (m*unitk[0]);
#    if sqk <= gsqmx:
#      kxvecs.append(m)
#      kyvecs.append(0)
#      kzvecs.append(0)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kcount += 1
#    sqk = (m*unitk[1]) * (m*unitk[1]);
#    if sqk <= gsqmx:
#      kxvecs.append(0)
#      kyvecs.append(m)
#      kzvecs.append(0)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kcount += 1
#    sqk = (m*unitk[2]) * (m*unitk[2]);
#    if sqk <= gsqmx:
#      kxvecs.append(0)
#      kyvecs.append(0)
#      kzvecs.append(m)
#      ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#      kcount += 1

#  # 1 = (k,l,0), 2 = (k,-l,0)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
#      if sqk <= gsqmx:
#        kxvecs.append(k)
#        kyvecs.append(l)
#        kzvecs.append(0)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1
#        
#        kxvecs.append(k)
#        kyvecs.append(-l)
#        kzvecs.append(0)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1

#  # 1 = (0,l,m), 2 = (0,l,-m)

#  for l in range(1,ny+1):
#    for m in range(1,nz+1):
#      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m)
#      if sqk <= gsqmx:
#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1

#        kxvecs.append(0)
#        kyvecs.append(l)
#        kzvecs.append(-m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1

#  # 1 = (k,0,m), 2 = (k,0,-m)

#  for k in range(1,nx+1):
#    for m in range(1,nz+1):
#      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m)
#      if sqk <= gsqmx:
#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1

#        kxvecs.append(k)
#        kyvecs.append(0)
#        kzvecs.append(-m)
#        ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#        kcount += 1

#  # 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

#  for k in range(1,nx+1):
#    for l in range(1,ny+1):
#      for m in range(1,nz+1):
#        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
#        if sqk <= gsqmx:
#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kcount += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kcount += 1

#          kxvecs.append(k)
#          kyvecs.append(l)
#          kzvecs.append(-m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kcount += 1

#          kxvecs.append(k)
#          kyvecs.append(-l)
#          kzvecs.append(-m)
#          ug.append(preu*np.exp(-0.25*sqk*alphasqinv)/sqk)
#          kcount += 1




































