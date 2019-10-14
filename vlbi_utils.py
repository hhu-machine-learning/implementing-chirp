import numpy as np

def ftVectors(uvcoord, fov, scale, pulse_ft):
    
    """
    Calculate the Fourier transform vectors from Eq.(7).
    
    Inputs:
    - uvcoord: A numpy array of shape (M,2) contaning M uv-coordinates.
    - fov: Field of view (float).
    - scale: The number of pulse functions used in the continuous image 
             representation (int).
    - pulse_ft: Closed-form Fourier transform of the pulse function used in 
                the continuous image representation.    
    Outputs:
    - gammas: A numpy array of shape (M,scale**2) containing the Fourier 
              transform vectors from Eq.(7) corresponding to the 
              uv-coordinates.
    """
    
    delta = fov/scale
    shift = delta * np.arange(scale) + delta/2 - fov/2
    # Eq.(7)
    gammas = np.array([pulse_ft(u, v, delta) *\
                       np.outer(np.exp(-2j*np.pi * v*shift), 
                                np.exp(-2j*np.pi * u*shift)
                                ).flatten(order='F')
                       for u, v in uvcoord])
    
    return gammas

def initImage(data, fov, scale, pulse_ft):

    """
    Initialize  the image coefficients as the dirty image.
    
    Inputs:
    - data: A dictionary containing the necessary data from OIFITS.
    - fov: Field of view (float).
    - scale: The number of pulse functions used in the continuous image 
             representation (int).
    - pulse_ft: Closed-form Fourier transform of the pulse function used in 
                the continuous image representation.       
    Outputs:
    - x: A numpy array of shape (scale,scale) contaning image coefficients.   
    """

    gamma = ftVectors(data['vis_uvcoord'], fov, scale, pulse_ft)
    gamma_conj = np.conjugate(gamma).T
    u = data['vis_uvcoord'][:,0]
    v = data['vis_uvcoord'][:,1]
    denom = pulse_ft(u,v,fov/scale) * pulse_ft(-u,-v,fov/scale) * scale**2 
    # Eq.(15)
    x = np.real(gamma_conj @ (data['vis']/denom))
    x = np.reshape(x, (scale,scale), order='F')
    
    x = x / np.sum(x)
    x = np.rot90(x,2)   # (init is upside down)
    
    return x

def upscaleImage(x, fov, new_scale, pulse):
    
    """
    Calculate the discretized continuous image representation.
    
    Inputs:
    - x: A numpy array of shape (scale,scale) contaning the image coefficients.
    - fov: Field of view (float).
    - new_scale: Size of the continuous image representation's discretization 
                 (int).
    - pulse: Pulse function used in the continuous image representation.   
    Outputs:
    - im: A numpy array of shape (new_scale,new_scale) contaning the 
          (discretized) continuous image representation.
    """
    
    scale = x.shape[0]
    delta = fov/scale
    new_delta = fov/new_scale
    
    # Eq.(6)
    cir = lambda l, m: np.sum([[x[j,i] *\
                                pulse(l - (delta*i + delta/2 - fov/2), 
                                      m - (delta*j + delta/2 - fov/2), delta) 
                                for j in range(scale)] 
                                for i in range(scale)], axis=(0,1))

    new_shift = new_delta * np.arange(new_scale) + new_delta/2 - fov/2
    ll, mm = np.meshgrid(new_shift, new_shift)
    im = cir(ll,mm)       # measured in Jy
    
    im *= new_delta**2    # measured in Jy / pixel
    
    return im

def mostLikelyPatches(x, beta, data, patch_size, gmm):

    """
    Optimize the cost function in terms of Z to create an image prior from 
    generated patch priors.
    
    Inputs:
    - x: A numpy array of shape (N,N) contaning the current image coefficients.
    - gmm: A dictionary containing the GMM's parameters.
    - patch_size: Size of the patches (int).
    - beta: half quadratic splitting's weighting parameter (int).        
    Outputs:
    - Z: A numpy array of shape (N,N) contaning the image prior.  
    """
    
    # normalize the image coefficients (inspired by patch_prior.py 
    # line 50-53 of https://github.com/achael/eht-imaging/blob/
    # 960a79557b4de7f2776bcfa1aef2c37cea487ab7/patch_prior.py)
    minVal = np.min(x)
    maxVal = np.max(x)
    x = (x - minVal) / maxVal
    
    xPad = np.pad(x , (patch_size-1, patch_size-1), 'constant')
    patches = extractPatches(xPad, patch_size)
    meanPatches =  np.mean(patches, axis=0)
    patches -= np.tile(meanPatches, (patch_size**2, 1)) 
    
    # determine the mixture components
    w_scores = np.zeros((gmm['n_components'], patches.shape[1]))
    for i in range(gmm['n_components']):
        # add noise to covariance matrices (inspired by patch_prior.py 
        # line 116 of https://github.com/achael/eht-imaging/blob/
        # 960a79557b4de7f2776bcfa1aef2c37cea487ab7/patch_prior.py)
        G = np.linalg.cholesky(gmm['covs'][:,:,i] +\
                               1/beta * np.eye(patch_size**2))
        # Eq.(16)
        w_scores[i,:] = np.log(gmm['weights'][i])            -\
                        patches.shape[0]/2 * np.log(2*np.pi) -\
                        np.log(np.prod(np.diagonal(G)))      -\
                        1/2 * np.sum((np.linalg.inv(G) @ patches)**2 , axis=0)
    j_star = w_scores.argmax(axis = 0)

    # create a set of auxiliary patches
    Zs = np.zeros(patches.shape)
    for j in range(gmm['n_components']):
        inds = np.where(j_star == j)[0]
        # Eq.(17)
        Zs[:,inds] = np.linalg.inv(gmm['covs'][:,:,j] +\
                                  1/beta * np.eye(patch_size**2)) @\
                   (gmm['covs'][:,:,j] @ patches[:,inds] +\
                    1/beta * np.tile(gmm['means'][:,j], (len(inds), 1)).T)
    Zs += np.tile(meanPatches, (patch_size**2, 1))
    
    # combine the auxiliary patches into a prior image
    domainPad = np.pad(np.ones_like(x), 
                       (patch_size-1, patch_size-1), 
                       'constant')
    inds_patches = extractPatches(np.reshape(range(np.prod(domainPad.shape)), 
                                             domainPad.shape, order='F'), 
                                  patch_size)
    stackedZ = np.bincount(inds_patches.flatten(order='F').astype(np.int), 
                           weights=Zs.flatten(order='F'))
    Z_vec = np.extract(domainPad.flatten(order='F'), stackedZ)
    Z_vec /= (patch_size**2)
    
    Z_vec = (Z_vec * maxVal) + minVal
    # set negative entries to zero (inspired by patch_prior.py 
    # line 78 of https://github.com/achael/eht-imaging/blob/
    # 960a79557b4de7f2776bcfa1aef2c37cea487ab7/patch_prior.py)
    Z_vec[Z_vec < 0] = 0
    # normalize prior to maximum absolute visbility (inspired by 
    # linearize_energy.py line 17-18 of https://github.com/achael/eht-imaging/
    # blob/960a79557b4de7f2776bcfa1aef2c37cea487ab7/linearize_energy.py
    Z_vec = np.max(np.abs(data['vis'])) * Z_vec / np.sum(Z_vec)

    Z = np.reshape(Z_vec, x.shape, order='F')
               
    return Z

def extractPatches(im, patch_size):
    
    """
    Extract all the possible patches from an image.
    
    Inputs:
    - im: A numpy array of shape (N,N) contaning an image with scalar data.
    - patch_size: Size of the patches to be extracted (int).      
    Outputs:
    - patches: A numpy array of shape (P,M) containing the extracted 
               vectorized patches in columns.
    """
    
    N = im.shape[0]
    P = patch_size**2
    M = (N - patch_size + 1)**2
    patches = np.zeros((P, M))
    for i in range(M):
        row = i % (N - patch_size + 1)
        col = i // (N - patch_size + 1)
        patches[:,i] = np.reshape(im[row:row+patch_size, col:col+patch_size], 
                                  P, order='F')
    return patches

def taylorExpansion(x, Z, beta, data, gammas, patch_size, lam):

    """
    Optimize the cost function in terms of x performing a second order 
    Taylor expansion.
    
    Inputs:
    - x: A numpy array of shape (N,N) contaning the current image coefficients.
    - Z: A numpy array of shape (N,N) contaning the current image prior.  
    - beta: half quadratic splitting's weighting parameter (int).
    - data: A dictionary containing the necessary data from OIFITS.
    - gammas: A triple of numpy arrays containing the Fourier transform vectors 
              from Eq. 5 corresponding to the uv-coordinates from the 
              bispectrum measurements.
    - patch_size: Size of the patches (int).
    - lam: Weighting parameter of the data term (float).
    Outputs:
    - x: A numpy array of shape (N,N) contaning the new image coefficients.   
    """
    
    x_vec = x.flatten(order='F')
    Z_vec = Z.flatten(order='F')
    
    gr1, gr2, gr3 = np.real(gammas)
    gi1, gi2, gi3 = np.imag(gammas)
    gr1x, gr2x, gr3x = gr1 @ x_vec, gr2 @ x_vec, gr3 @ x_vec
    gi1x, gi2x, gi3x = gi1 @ x_vec, gi2 @ x_vec, gi3 @ x_vec
    # Eq.(20)
    rXi = gr1x*gr2x*gr3x - gi1x*gi2x*gr3x -\
          gr1x*gi2x*gi3x - gi1x*gr2x*gi3x 
    iXi = gr1x*gi2x*gr3x + gi1x*gr2x*gr3x +\
          gr1x*gr2x*gi3x - gi1x*gi2x*gi3x
    rMeas = np.real(data['bi'])
    iMeas = np.imag(data['bi'])
    
    # Eq.(21)
    rA = gr1 * (gr2x * gr3x)[:,np.newaxis]  +\
         gr2 * (gr1x * gr3x)[:,np.newaxis]  +\
         gr3 * (gr1x * gr2x)[:,np.newaxis]  -\
        (gi1 * (gi2x * gr3x)[:,np.newaxis]  +\
         gi2 * (gi1x * gr3x)[:,np.newaxis]  +\
         gr3 * (gi1x * gi2x)[:,np.newaxis]) -\
        (gr1 * (gi2x * gi3x)[:,np.newaxis]  +\
         gi2 * (gr1x * gi3x)[:,np.newaxis]  +\
         gi3 * (gr1x * gi2x)[:,np.newaxis]) -\
        (gi1 * (gr2x * gi3x)[:,np.newaxis]  +\
         gr2 * (gi1x * gi3x)[:,np.newaxis]  +\
         gi3 * (gi1x * gr2x)[:,np.newaxis])
    # Eq.(22)
    iA = gr1 * (gi2x * gr3x)[:,np.newaxis]  +\
         gi2 * (gr1x * gr3x)[:,np.newaxis]  +\
         gr3 * (gr1x * gi2x)[:,np.newaxis]  +\
         gi1 * (gr2x * gr3x)[:,np.newaxis]  +\
         gr2 * (gi1x * gr3x)[:,np.newaxis]  +\
         gr3 * (gi1x * gr2x)[:,np.newaxis]  +\
         gr1 * (gr2x * gi3x)[:,np.newaxis]  +\
         gr2 * (gr1x * gi3x)[:,np.newaxis]  +\
         gi3 * (gr1x * gr2x)[:,np.newaxis]  -\
        (gi1 * (gi2x * gi3x)[:,np.newaxis]  +\
         gi2 * (gi1x * gi3x)[:,np.newaxis]  +\
         gi3 * (gi1x * gi2x)[:,np.newaxis])
    rB = rXi - rA @ x_vec
    iB = iXi - iA @ x_vec
    
    factor = 3/data['num_tele']/(data['bi_amperr']**2)
    
    sum1 = (factor[:,np.newaxis] * rA).T @ rA +\
           (factor[:,np.newaxis] * iA).T @ iA
    sum2 = (factor[:,np.newaxis] * rA).T @ (rB - rMeas) +\
           (factor[:,np.newaxis] * iA).T @ (iB - iMeas)
          
    # Eq.(19)
    new_x_vec = np.linalg.solve(
                lam * sum1 + beta * (patch_size**2) * np.eye(len(Z_vec)),
                - lam * sum2 + beta * (patch_size**2) * Z_vec)
   
    new_x = np.reshape(new_x_vec, x.shape, order='F')
    
    return new_x

def getData(oidata):
    
    """
    Extract the necessary data from OIFITS. 
    
    Inputs:
        - oidata: OIFITS data format extraced by Paul Boley's OIFITS module 
                  containing visibilities and bispectrum measurements.
    Ouputs:
        - data: A dictionary containing the necessary data from OIFITS.
    """
    
    data = {}
    
    wav = oidata.wavelength['WAVELENGTH_NAME'].eff_wave[0]
    data['vis_uvcoord'] = np.array([[v.ucoord/wav, 
                                     v.vcoord/wav] for v in oidata.vis])
    vis_amp = np.array([v.visamp[0] for v in oidata.vis])
    vis_phi = np.array([v.visphi[0] for v in oidata.vis]) * np.pi/180
    data['vis'] = vis_amp * np.exp(1j * vis_phi)
    data['vis_amperr'] = np.array([v.visamperr[0] for v in oidata.vis])
        
    data['bi_uvcoord1'] = np.array([[t3.u1coord/wav, t3.v1coord/wav] 
                                    for t3 in oidata.t3])
    data['bi_uvcoord2'] = np.array([[t3.u2coord/wav, t3.v2coord/wav] 
                                    for t3 in oidata.t3])
    data['bi_uvcoord3'] = np.array([[-(t3.u1coord+t3.u2coord)/wav, 
                                     -(t3.v1coord+t3.v2coord)/wav] 
                                    for t3 in oidata.t3])
        
    bi_amp = np.array([t3.t3amp[0] for t3 in oidata.t3])
    bi_phi = np.array([t3.t3phi[0] for t3 in oidata.t3]) * np.pi/180
    data['bi'] = bi_amp * np.exp(1j * bi_phi)
    data['bi_amperr'] = np.array([t3.t3amperr[0] for t3 in oidata.t3])
    
    stations = {}
    for k in oidata.vis:
        t = k.timeobs.time()
        time = (t.hour * 60 + t.minute) * 60 + t.second
        stations[time] = []
    times = []
    for k in oidata.vis:
        t = k.timeobs.time()
        time = (t.hour * 60 + t.minute) * 60 + t.second
        stations[time].append(k.station[0].sta_name)
        stations[time].append(k.station[1].sta_name)
        times.append((t.hour * 60 + t.minute) * 60 + t.second)
    num_tele = []
    for t3 in oidata.t3:
        t = t3.timeobs.time()
        time = (t.hour * 60 + t.minute) * 60 + t.second
        unique_stations = np.unique(stations[time])
        num_tele.append(len(unique_stations))
    data['num_tele'] = np.array(num_tele)
        
    return data