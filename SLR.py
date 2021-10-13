import numpy as np
import utils
import numba

class hankel:
    """Basic Hankel Matrix Class"""
    
    @staticmethod
    @numba.njit(parallel=True)
    def fwd(x, kernel):

        # Reshape to have coil dimension as dim2
        #x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        nc = x.shape[2]
        
        # Get kernel-reduced dimensions
        #dimr = tuple(map(lambda i,j:i-j+1, x.shape[:2], kernel))
        dimr = (x.shape[0]-kernel[0]+1, x.shape[1]-kernel[1]+1)
        
        # Initialise matrix components
        h = np.zeros((np.prod(np.array(dimr)), np.prod(np.array(kernel)), nc), dtype=np.complex_)
        
        # Loop over all kernel locs
        for kx in numba.prange(dimr[0]):
            for ky in numba.prange(dimr[1]):
                h[kx*dimr[1]+ky,...] = x[kx:kx+kernel[0],ky:ky+kernel[1],:].copy().reshape((1,-1,nc))
                
        return h
    
    @staticmethod
    @numba.njit
    def adj(h, dims, kernel):

        # get coil dimension
        nc = h.shape[2]
        
        # get kernel-reduced dimensions
        #dimr = tuple(map(lambda i,j:i-j+1, dims, kernel))
        dimr = (dims[0]-kernel[0]+1, dims[1]-kernel[1]+1)

        # initialise output
        x = np.zeros((dims[0], dims[1], nc), dtype=np.complex_)
        
        # loop over all kernel locs
        for kx in range(dimr[0]):
            for ky in range(dimr[1]):
                x[kx:kx+kernel[0],ky:ky+kernel[1],:] += h[kx*dimr[1]+ky,...].reshape((kernel[0], kernel[1], nc))
        
        return x
    
    @staticmethod
    def norm(dims, kernel):
        
        if dims[0] >  2*kernel[0]:
            U = np.concatenate((np.arange(kernel[0])+1, kernel[0]*np.ones((dims[0]-2*kernel[0])), np.arange(kernel[0],0,-1)))
        elif dims[0] > kernel[0]:
            U = np.arange((dims[0]+1)//2)+1
            U = np.concatenate((U[:(dims[0]+1)//2],np.flip(U[:(dims[0])//2])))
        else:
            U = np.ones((kernel[0],))
        if dims[1] > 2*kernel[1]:
            V = np.concatenate((np.arange(kernel[1])+1, kernel[1]*np.ones((dims[1]-2*kernel[1])), np.arange(kernel[1],0,-1)))
        elif dims[1] > kernel[1]:
            V = np.arange((dims[1]+1)//2)+1
            V = np.concatenate((V[:(dims[0]+1)//2],np.flip(V[:(dims[0])//2])))
        else:
            V = np.ones((kernel[1],))
            
        N = U[:,np.newaxis]@V[np.newaxis,:]
        
        return N[:,:,np.newaxis]

    @staticmethod
    def size(dims, kernel):
        return ((dims[0]-kernel[0]+1)*(dims[1]-kernel[1]+1), np.prod(kernel))
    
class c_matrix:
    """LORAKS-style C-Matrix"""
    
    @staticmethod
    def fwd(x, kernel):

        c = hankel.fwd(x, kernel)
        
        return c.reshape((c.shape[0],-1))
        
    @staticmethod
    def adj(c, dims, kernel):
        
        c = c.reshape(np.concatenate((c_matrix.size(dims,kernel),(-1,))))
        
        return hankel.adj(c, dims, kernel)

    @staticmethod
    def norm(dims, kernel):
        return hankel.norm(dims, kernel)

    @staticmethod
    def size(dims, kernel):
        return hankel.size(dims, kernel)
    

class s_matrix:
    """LORAKS-style S-Matrix"""
    
    @staticmethod
    def fwd(x, kernel):

        s_pos = hankel.fwd(x, kernel)
        s_neg = np.flip(s_pos, axis=0)
        
        s = np.concatenate((np.concatenate((np.real(s_pos)-np.real(s_neg), 
                                            np.imag(s_neg)-np.imag(s_pos)),axis=1),
                            np.concatenate((np.imag(s_pos)+np.imag(s_neg), 
                                            np.real(s_pos)+np.real(s_neg)),axis=1)),axis=0)
        return s.reshape((s.shape[0],-1))

        
    @staticmethod
    def adj(s, dims, kernel):

        s = s.reshape(np.concatenate((s_matrix.size(dims,kernel),(-1,))))

        A = s[:s.shape[0]//2, :s.shape[1]//2, :]
        B = s[:s.shape[0]//2, s.shape[1]//2:, :]
        C = s[s.shape[0]//2:, :s.shape[1]//2, :]
        D = s[s.shape[0]//2:, s.shape[1]//2:, :]

        s_pos = A+D + 1j*(C-B)
        s_neg = D-A + 1j*(B+C)

        s_pos = s_pos + np.flip(s_neg, axis=0)

        return hankel.adj(s_pos, dims, kernel)
    
    @staticmethod
    def norm(dims, kernel):
        return 4*c_matrix.norm(dims, kernel)
    
    @staticmethod
    def size(dims, kernel):
        return c_matrix.size(dims, kernel)*np.array([2,2])

class vcc_matrix:
    """Virtual conjugate channel matrix, alternative formulation of S-Matrix"""
    
    @staticmethod
    def fwd(x, kernel):

        v_pos = hankel.fwd(x, kernel)
        v_neg = np.conj(np.flip(v_pos))
        
        v = np.concatenate((v_pos, v_neg), axis=1)
        
        return v.reshape((v.shape[0],-1))
        
    @staticmethod
    def adj(v, dims, kernel):
        
        v = v.reshape(np.concatenate((vcc_matrix.size(dims,kernel),(-1,))))

        v_pos = v[:, :v.shape[1]//2, :]
        v_neg = v[:, v.shape[1]//2:, :]

        v_pos = v_pos + np.flip(np.conj(v_neg))
        
        return hankel.adj(v_pos, dims, kernel)
    
    @staticmethod
    def norm(dims, kernel):
        return 2*hankel.norm(dims, kernel)
    
    @staticmethod
    def size(dims, kernel):
        return hankel.size(dims, kernel)*np.array([1,2])
    
class xi_matrix:
    """Virtual conjugate channel matrix, alternative formulation of S-Matrix"""
    
    @staticmethod
    def fwd(x, kernel):

        v_pos = hankel.fwd(x, kernel)
        v_neg = hankel.fwd(np.roll(np.conj(np.flip(x)),(1,1),(0,1)), kernel)
        
        v = np.concatenate((v_pos, v_neg), axis=1)
        
        return v.reshape((v.shape[0],-1))
        
    @staticmethod
    def adj(v, dims, kernel):
        
        v = v.reshape(np.concatenate((vcc_matrix.size(dims,kernel),(-1,))))

        v_pos = v[:, :v.shape[1]//2, :]
        v_neg = v[:, v.shape[1]//2:, :]

        return hankel.adj(v_pos, dims, kernel) + np.flip(np.conj(np.roll(hankel.adj(v_neg, dims, kernel), (-1,-1),(0,1))))
    
    @staticmethod
    def norm(dims, kernel):
        N = hankel.norm(dims, kernel)
        return N + np.roll(N, (1,1), (0,1))
    
    @staticmethod
    def size(dims, kernel):
        return hankel.size(dims, kernel)*np.array([1,2])

def h_fwd(x, kernel):

        # Reshape to have coil dimension as dim2
        #x = np.reshape(x, (x.shape[0], x.shape[1], -1))
    nc = x.shape[2]
        
        # Get kernel-reduced dimensions
        #dimr = tuple(map(lambda i,j:i-j+1, x.shape[:2], kernel))
    dimr = (x.shape[0]-kernel[0]+1, x.shape[1]-kernel[1]+1)
        
        # Initialise matrix components
    h = np.zeros((np.prod(np.array(dimr)), np.prod(np.array(kernel)), nc), dtype=np.complex_)
        
        # Loop over all kernel locs
    for kx in numba.prange(dimr[0]):
        for ky in numba.prange(dimr[1]):
            h[kx*dimr[1]+ky,...] = x[kx:kx+kernel[0],ky:ky+kernel[1],:].copy().reshape((1,-1,nc))
                
    return h
def maj_min(d, mtx, kernel, lambda_s, r, batch=None, niters=100, tol=1E-4, gt=None, init=None):
    """Majorize Minimize Reconstruction"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    z = init
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    P = 1/(M + lambda_s*N)
    P[np.isinf(P)] = 0
    
    if batch is None:
        def batch(dims, kernel, mtx):
            return np.arange(mtx.size(dims, kernel)[0]).reshape((1,-1))
    
    # Majorize-Minimize iterations
    for i in range(niters):
        
        # Get structured matrix
        H = mtx.fwd(z, kernel)
        
        # Truncate rank
        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(H[idx,:])
            H[idx,:] = H[idx,:]@(V[:,:r]@np.conj(V[:,:r].T))
            
        # Update estimate
        zz = P*(d + lambda_s*mtx.adj(H, dims, kernel))
        
        # Check relative update tolerance
        update = np.linalg.norm(zz.ravel()-z.ravel())/np.linalg.norm(z.ravel())
        if update < tol:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(zz[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
 
        # Save estimate
        z = zz
    
    if gt is not None:
        return __unpad(z, crop), err
    else:
        return __unpad(z, crop)
        
def ADMM(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)


        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(H[idx,:] + u[idx,:])
            z[idx,:] = (H[idx,:] + u[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)
def ADMM_log(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)

        tem = H + u
        tem=np.log(tem+np.finfo(float).eps)

        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(tem[idx,:])
            z[idx,:] = (tem[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
            
        z=np.exp(z)
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)


def ADMM_centralk(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)
        _,V = utils.half_SVD(H[2511-200:2672+400,:] + u[2511-200:2672+400,:])

        for idx in batch(dims, kernel, mtx):
#             _,V = utils.half_SVD(H[idx,:] + u[idx,:])
            z[idx,:] = (H[idx,:] + u[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)


def ADMM_centralV(d, mtx, kernel, r, Nb, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)

        nn=np.zeros((Nb))
        idx=batch(dims, kernel, mtx)
#         print(idx.shape)
        for ii in range(Nb):
            nn[ii] = np.linalg.norm(H[idx[ii,:],:] + u[idx[ii,:],:])
            
        vidx=np.argmax(nn) 
        _,V = utils.half_SVD(H[idx[vidx,:],:] + u[idx[vidx,:],:])
        
        for iii in range(Nb):
#             _,V = utils.half_SVD(H[idx,:] + u[idx,:])
            z[idx[iii,:],:] = (H[idx[iii,:],:] + u[idx[iii,:],:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)

def ADMM_normH(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
#     print(M.shape)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
#     print(z.shape)

    L = mtx.size(dims, kernel)[0]
    fac=np.ones(z.shape)
    gt_p, _ = __pad(gt)
    H_gt = mtx.fwd(gt_p, kernel)
    for ig in range(L):
        fac[ig,:]=1/np.linalg.norm(H_gt[ig,:])
#         fac[ig,:]=np.abs(ig-L//2)*np.abs(ig-L//2)
#     print(fac.shape)
        
    # Get normalisation factor
#     N = mtx.norm(dims, kernel)
    
    N=mtx.adj(fac*fac*mtx.fwd(np.ones(x.shape), kernel),dims,kernel)
#     print(N.shape)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(fac*(z - u), dims, kernel))
        
        # z-update
        H = fac*mtx.fwd(xx, kernel)


        tem = H + u

            
#         print(fac.shape)
        
        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(tem[idx,:])
            z[idx,:] = (tem[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
     
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)



def ADMM_norm(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    fac=np.zeros((1,L))
    
    gt_p, _ = __pad(gt)
    H_gt = mtx.fwd(gt_p, kernel)
#     for ig in range(L):
#         fac[0,ig]=np.linalg.norm(H_gt[ig,:])
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)


        tem = H + u
        for ii in range(L):
            fac[0,ii]=np.linalg.norm(tem[ii,:])
            tem[ii,:]=tem[ii,:]/fac[0,ii]
            
#         print(fac.shape)
        
        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(tem[idx,:])
            z[idx,:] = (tem[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        
        for ii in range(L):
            z[ii,:]=z[ii,:]*fac[0,ii]
        
        
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)

def ADMM_normgt(d, mtx, kernel, r, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    fac=np.zeros((1,L))
    
    gt_p, _ = __pad(gt)
    H_gt = mtx.fwd(gt_p, kernel)
    for ig in range(L):    
        fac[0,ig]=np.linalg.norm(H_gt[ig,:])
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
#         idx=np.mod(np.arange(np.int(np.floor(L*0.2)))+np.int(np.random.random()*L),L)
#         print(idx)


        tem = H + u
        for ii in range(L):
#             fac[0,ii]=np.linalg.norm(tem[ii,:])
            tem[ii,:]=tem[ii,:]/fac[0,ii]
            
#         print(fac.shape)
        
        for idx in batch(dims, kernel, mtx):
            _,V = utils.half_SVD(tem[idx,:])
            z[idx,:] = (tem[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        
        for ii in range(L):
            z[ii,:]=z[ii,:]*fac[0,ii]
        
        
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)



def ADMM_soft(d, mtx, kernel, l, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)

        for idx in batch(dims, kernel, mtx):
            S,V = utils.half_SVD(H[idx,:] + u[idx,:])
#             print(S)
#             print(S.shape)
#             print(np.maximum(S-l/p,0))
#             print(np.maximum(S-l/p,0).shape)
            S=np.diag(S)
            U=(H[idx,:] + u[idx,:])@V@np.linalg.pinv(S)
            S=np.maximum(S-l/p,0)
            z[idx,:] = U@S@np.conj(V.T)
        
        # u-update
        u = u + H - z
        
        #penalty adjustment
        s  = p*mtx.fwd(xx-x, kernel)
        if np.linalg.norm(z-H) > 10*np.linalg.norm(s):
            p = p*2
            u = u/2
        elif np.linalg.norm(s) >10*np.linalg.norm(z-H):
            p = p/2
            u = u*2
            
#         print(p)
                
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, P: {p}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)


    
def ADMMalter(d, mtx, kernel, r, mode=1, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
        idx = batch(dims, kernel, mtx)
#         print(idx.shape)
        if mode ==1:
            ii=np.mod(i,2)
        else:
            ii=0
        _,V = utils.half_SVD(H[idx[ii,:],:] + u[idx[ii,:],:])
        z[idx[ii,:],:] = (H[idx[ii,:],:] + u[idx[ii,:],:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)
    
def ADMM_regular(d, mtx, kernel, r, num, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)
        print(int(np.floor(0.3*L//num*i)))
        for idx in np.mod(batch(dims, kernel, mtx) + int(np.floor(0.618*L//num*i)),L):
            print(idx.shape)
            _,V = utils.half_SVD(H[idx,:] + u[idx,:])
            z[idx,:] = (H[idx,:] + u[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)
   

def ADMM_single(d, mtx, kernel, r, ratio, batch=None, p=1E-6, niters=100, tol=1E-4, gt=None, init=None):
    """Non-convex ADMM reconstruction with strict rank constraints"""
    
    # Pad input
    d, crop = __pad(d)
    dims = d.shape[:2]
    
    # Get sampling mask
    M = (d !=0)
    
    # Initialise 
    if init is None:
        init = d
    else:
        init, _ = __pad(init)
        
    if batch is None:
        batch = sub_block(1)
        
    if gt is not None:
        err = np.zeros(niters)
    else:
        err = []
        
    x = init
    z = mtx.fwd(x, kernel)
    u = 0*z
    
    # Get normalisation factor
    N = mtx.norm(dims, kernel)
    
    # Precompute LHS
    Q = 1/(M + (p/2)*N)
    Q[np.isinf(Q)] = 0
    
    L = mtx.size(dims, kernel)[0]
    
    # ADMM iterations
    for i in range(niters):
        
        # x-update
        xx = Q*(d + (p/2)*mtx.adj(z - u, dims, kernel))
        
        # z-update
        H = mtx.fwd(xx, kernel)

        idx = np.mod(np.arange(np.int(np.floor(L*ratio)))+np.int(np.random.random()*L),L)

        _, V = utils.half_SVD(H[idx,:] + u[idx,:])
        z[idx,:] = (H[idx,:] + u[idx,:])@(V[:,:r]@np.conj(V[:,:r].T))
        
        # u-update
        u = u + H - z
        
        # Check relative update tolerance
        update = np.linalg.norm(xx.ravel()-x.ravel())/np.linalg.norm(x.ravel())
        if update < tol and i > 0:
            print(f'Min Update Tolerance Reached at {i} iterations')
            break
            
            
        # If ground truth available, print RMSE
        if gt is not None:
            err[i] = np.linalg.norm(xx[:crop[0],:crop[1]].ravel()-gt.ravel())/np.linalg.norm(gt.ravel())
            if np.mod(i,100) == 0:
                print(f'Iter: {i}, RMSE: {err[i]}')
        
        # Save estimate
        x = xx 

    if gt is not None:
        return __unpad(x, crop), err
    else:
        return __unpad(x, crop)
   
# contiguous submatrix blocks
def sub_block(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
        return np.mod(np.arange((L//N)*N)+np.int(np.random.random()*L),L).reshape((N,-1))
        
    return fn

def sub_blockS(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
#         print(L)
        L=L//2
#         tem=np.zeros([(L//N)*N]).reshape((N,-1))
        tem1 = np.mod(np.arange((L//N)*N)+np.int(np.random.random()*L),L).reshape((N,-1))
        tem2 = tem1+L

        return np.concatenate((tem1, tem2), axis=1)
#         return np.concatenate((np.mod(np.arange((L//N)*N),L).reshape((N,-1)), np.mod(np.arange((L//N)*N),L).reshape((N,-1))+L), axis=1)
        
    return fn

def sub_blockfix(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
        return np.mod(np.arange((L//N)*N),L).reshape((N,-1))
        
    return fn



def sub_blockoverlap(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
        ratio=0.3
#         overlapping width (L//N)*ratio
        LL=np.int(np.floor((L//N)*ratio))*(N-1)+L
        inc=np.int(np.floor((L//N)*ratio))
        temp=np.mod(np.arange(((L//N)+inc)*N)+np.int(np.random.random()*L),L).reshape((N,-1))
        for i in range(N):
            temp[i,:]=np.mod(temp[i,:]-i*inc,L)
        return temp
        
    return fn

# interleaved rows submatrix "stripes"
def sub_stripe(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
        #return np.mod(np.arange((L//N)*N),L).reshape((-1,N)).T
        return [np.arange(x,L,N) for x in range(N)]
        
    return fn

# random row submatrices
def sub_rand(N):
    def fn(dims, kernel, mtx):
        L = mtx.size(dims, kernel)[0]
        return np.random.permutation((L//N)*N).reshape((N,-1))
    
    return fn
    
## Helper
def __pad(d):
    # Zero-pad input if even, to make k-space symmetric about origin
    # Also add coil dimension if not present
    if np.ndim(d) == 2:
        d = d[:,:,np.newaxis]
        
    pad = 1 - np.mod(d.shape[:2],2)
    d = np.pad(d, ((0,pad[0]),(0,pad[1]),(0,0)), mode='wrap')
    crop = d.shape[:2] - pad
    
    return d, crop

def __unpad(x, crop):
    if x.shape[2] == 1:
        return x[:crop[0],:crop[1],0]
    else:
        return x[:crop[0],:crop[1],:]