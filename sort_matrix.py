from numpy import *

A = loadtxt("amatrix")

tags = array(A[0,:],dtype=int)

A = A[1:,:]

n = shape(A)
n = n[0]

# sort columns first
idxs = argsort(tags)
A = A[idxs,:]

# loop over rows of matrix
f = open("amatrix_sorted","w")
f.write(' '.join(map(str,tags[idxs]))+'\n')
for i in range(n):
    
  # shift idxs to right
  nidxs = roll(idxs, i)
  
  # sort each row
  f.write(' '.join(map(str,roll(A[i,nidxs],-i)))+'\n')

  
f.close()
