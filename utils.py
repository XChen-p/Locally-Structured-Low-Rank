import numpy as np

def half_SVD(x):
    d, v = np.linalg.eig(np.conj(x).T@x)
    
    ii = np.argsort(np.abs(d))
    
    s = np.sqrt(d[ii[::-1]])
    v = v[:,ii[::-1]] 
    
    return s, v

def fftdim(x, dims=None):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), axes=dims, norm="ortho"))

def ifftdim(x, dims=None):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x), axes=dims, norm="ortho"))

def noise(dims, std=1):
    return std*(np.random.standard_normal(dims)+1j*np.random.standard_normal(dims))/np.sqrt(2)

def sos(x, axis=-1):
    return np.sqrt(np.sum(np.abs(x)**2,axis=axis))


def poisson_disc(N,r):
    mask = np.zeros((N,N))-1
    active = []
    
    # define r->2r annulus
    tmp = np.zeros((np.int(4*r),np.int(4*r)))
    for x in np.arange(np.int(4*r)):
        for y in np.arange(np.int(4*r)):
            if r < np.sqrt((x-2*r)**2 + (y-2*r)**2) < 2*r:
                tmp[x,y] = 1
                
    idx = np.where(tmp == 1)
    annulus = np.concatenate((idx[0][:,np.newaxis],idx[1][:,np.newaxis]),axis=1) - np.int(2*r)
    
    counter = 0
    mask[np.random.randint(N),np.random.randint(N)] = counter
    active.append(counter)
    
    while len(active) > 0:
        idx = np.random.randint(len(active))
        loc = np.where(mask == active[idx])
        
        x = np.mod(annulus + np.array([loc[0],loc[1]]).reshape((-1,2)),N)
        
        remove = True
        for i in np.random.permutation(len(x))[:30]:
            hit = False
            for rx in np.arange(np.int(-r),np.int(r+1)):
                for ry in np.arange(np.int(-r),np.int(r+1)):
                    if np.sqrt(rx**2 + ry**2) <= r:
                        if mask[np.mod(rx+x[i][0],N),np.mod(ry+x[i][1],N)] >= 0:
                            hit = True
                            break
                if hit:
                    break
            
            if not hit:
                counter += 1
                mask[x[i][0],x[i][1]] = counter
                active.append(counter)
                remove = False
                break
        
        if remove:
            active.remove(active[idx])
    
    return mask >= 0