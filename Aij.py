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
Lx = 31.
Ly = 31.*wirefac
Lz = 31.*slabfac
V = Lx*Ly*Lz
Vinv = 1./V

# k-space
nx = 4
ny = 4 
nz = 9
nmax = max([nx,ny,nz])

kprefac = 2*np.pi*np.array([1./Lx,1./Ly,1./Lz])

ug = []
kxvecs = []
kyvecs = []
kzvecs = []

alpha = 0.1780932
eta = 1.979

# set some constants
etasqr2 = eta/np.sqrt(2)
selfcorr = (np.sqrt(2)*eta-2.*alpha)/np.sqrt(np.pi)
alphasqinv = 1.0 / alpha**2
preu = 4.0*np.pi*Vinv

def main():
  
  global N
  N = len(r)
  
  # precompute k-space coeffs
  #ewald_coeffs()

  # allocate Aij
  A = np.zeros([N,N]) 

  for i in range(N):
    
    # self correction (only on diagonal)
    A[i,i] += selfcorr
    
    for j in range(i,N) if symflag else range(N):
      
      # slab or wire pbc distances
      dx = np.mod(r[i,0]-r[j,0], Lx*.5)
      dy = r[i,1]-r[j,1] if wirefac > 1. else np.mod(r[i,1]-r[j,1], Ly*.5)
      dz = r[i,2]-r[j,2]
      rij = np.array([dx,dy,dz])
      dij = np.linalg.norm(rij)
      
      # real-space contributions
      if (i != j) & (dij < Rc): A[i,j] += ( math.erfc(alpha*dij) - math.erfc(etasqr2*dij) ) / dij 
      
      # k-space contributions (see metalwalls doc)
      for u in range(-nx,nx+1):
        for v in range(-ny,ny+1):
          for w in range(-nz,nz+1):
            if u+v+w == 0: continue
            kx = kprefac[0]*u
            ky = kprefac[1]*v
            kz = kprefac[2]*w
            
            ksq = np.dot([kx,ky,kz],[kx,ky,kz])
            kdotr = np.dot([kx,ky,kz],rij)
            
            pref = 4.*np.pi*Vinv*np.exp(-ksq/(4.*alpha**2))/ksq
          
            A[i,j] += pref*np.cos(kdotr)
      
#      kx = 0.
#      for v in range(1,ny+1):
#        for w in range(-nz,nz+1): 
#          ky = kprefac[1]*v
#          kz = kprefac[2]*w
#          
#          ksq = np.dot([kx,ky,kz],[kx,ky,kz])
#          kdotrj = np.dot([kx,ky,kz],r[j])
#          
#          # TODO: could be pre-computed in i-loop
#          Cx = np.cos(kx*r[i,0])
#          Cy = np.cos(ky*r[i,1])
#          Cz = np.cos(kz*r[i,2])
#          Sx = np.sin(kx*r[i,0])
#          Sy = np.sin(ky*r[i,1])
#          Sz = np.sin(kz*r[i,2])
#          
#          C = Cx*Cy*Cz - Sx*Cy*Cz - Cx*Sy*Cz - Cx*Cy*Sz
#          S = Sx*Cy*Cz + Cx*Sy*Cz + Cx*Cy*Sz - Sx*Sy*Sz
#          
#          pref = 8.0*np.pi*Vinv*np.exp(-ksq/(4.*alpha**2))/ksq
#          
#          A[i,j] += pref*(C*np.cos(kdotrj) + S*np.sin(kdotrj)) 
#          
#      for u in range(1,nx+1):
#        for v in range(-ny,ny+1):
#          for w in range(-nz,nz+1):
#            kx = kprefac[0]*u
#            ky = kprefac[1]*v
#            kz = kprefac[2]*w
#            
#            ksq = np.dot([kx,ky,kz],[kx,ky,kz])
#            kdotrj = np.dot([kx,ky,kz],r[j])
#            
#            Cx = np.cos(kx*r[i,0])
#            Cy = np.cos(ky*r[i,1])
#            Cz = np.cos(kz*r[i,2])
#            Sx = np.sin(kx*r[i,0])
#            Sy = np.sin(ky*r[i,1])
#            Sz = np.sin(kz*r[i,2])
#            
#            C = Cx*Cy*Cz - Sx*Cy*Cz - Cx*Sy*Cz - Cx*Cy*Sz
#            S = Sx*Cy*Cz + Cx*Sy*Cz + Cx*Cy*Sz - Sx*Sy*Sz
#            
#            pref = 8.0*np.pi*Vinv*np.exp(-ksq/(4.*alpha**2))/ksq
#            
#            A[i,j] += pref*(C*np.cos(kdotrj) + S*np.sin(kdotrj))  
      

      # slab or wire correction
      if wirefac > 1.: A[i,j] += 2.*np.pi*Vinv*(r[i,2]*r[j,2]+r[i,1]*r[j,1])
      else: A[i,j] += 4.*np.pi*Vinv*r[i,2]*r[j,2]

  if symflag: 
    # copy upper triangle to lower
    iu = np.triu_indices(N,1)
    il = (iu[1],iu[0])
    A[il]=A[iu]
    
  print(A)

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

if __name__ == "__main__":
    main()



































