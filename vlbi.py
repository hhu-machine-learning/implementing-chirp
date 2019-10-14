import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import vlbi_utils as utils
from astropy.io import fits
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from skimage.measure import compare_ssim

class CHIRP(object):
    
    def __init__(self, oidata, fov, naxis, **kwargs):
        
        """
        Inputs:
        - oidata: OIFITS data format extraced by Paul Boley's OIFITS module 
                  containing visibilities and bispectrum measurements.
        - fov: Field of view in radians (float).
        - naxis: Size of the source image (int).       
        Optional arguments:
        - pulse: Pulse function used in the continuous image representation.
                 Default is the two-dimensional triangular pulse.
        - pulse_ft: Closed-form Fourier transform of the pulse function used 
                    in the continuous image representation. Default is the 
                    Fourier transform of the two-dimensional triangular pulse.
        - gmm: A dictionary containing the mixture components of a pre-trained 
               Gaussian Mixture Model. Default a GMM trained on natural images
               taken from https://github.com/achael/eht-imaging/tree/960a79557b
               4de7f2776bcfa1aef2c37cea487ab7.
        - patch_size: Size of the patches (int) for the EPLL. Default is 8.
        - betas: A sorted list in ascending order of weighting parameters (int) 
                 for half quadratic splitting. Default is 
                 [1, 4, 8, 16, 32, 64, 128, 256, 512].
        - scales : A sorted list in ascending order of the number of pulse 
                   functions (int) in the continuous image representation. 
                   Default is [20, 23, 26, 29, 34, 38, 43, 49, 56, 64].
        """

        self.data = utils.getData(oidata)
        self.fov = fov
        self.naxis = naxis
        self.history = {}
        self.res = None
        self.__fig, self.__axs, self.__cbar = None, 2*[None], None

        pulse = lambda x, y, delta: \
                (np.maximum(1 - np.abs(x/delta), 0) / delta) *\
                (np.maximum(1 - np.abs(y/delta), 0) / delta)
        self.pulse = kwargs.pop('pulse', pulse)
        pulse_ft = lambda x, y, delta: \
                   np.sinc(x * delta)**2 *\
                   np.sinc(y * delta)**2
        self.pulse_ft = kwargs.pop('pulse_ft', pulse_ft)
        pdata = io.loadmat('naturalPrior.mat')
        gmm = {}
        gmm['n_components'] = pdata['nmodels'][0][0]
        gmm['weights'] = pdata['mixweights'].flatten()
        gmm['covs'] = pdata['covs']
        gmm['means'] = pdata['means']
        self.gmm = kwargs.pop('gmm', gmm)
        self.patch_size = kwargs.pop('patch_size', 8)
        self.betas = kwargs.pop('betas', 
                                [1, 4, 8, 16, 32, 64, 128, 256, 512])
        self.scales = kwargs.pop('scales', 
                                 [20, 23, 26, 29, 34, 39, 44, 50, 56, 64])
        
        if len(kwargs) > 0:
            raise Exception('Unrecognized arguments.')  


    def reconstruct(self, lam = 0.0001, display=False):
        
        """
        Reconstruct the image given the bispectrum measurements.
        
        Inputs:
        - lambda: Weighting parameter of the data term (float). Default
                  is 0.0001.
        - display: Flag for displaying partial solutions. Default is False.       
        Outputs:
        - x: A numpy array of shape (N,N) contaning the reconstructed image 
             coefficients.   
        """
        
        self.history = {}
        
        x = utils.initImage(self.data, self.fov, self.scales[0], self.pulse_ft)
        
        if display:
            plt.ion()
            plt.show()
            self.__fig, self.__axs = plt.subplots(1, 2)
            plt.subplots_adjust(wspace=-0.1)
            self.__display(x, np.zeros_like(x), 'Initialization')
            
        self.history['init'] = x
        
        for scale in self.scales:

            x = utils.upscaleImage(x, self.fov, scale, self.pulse)
            
            gammas = (utils.ftVectors(self.data['bi_uvcoord1'], self.fov, 
                                       scale, self.pulse_ft),
                      utils.ftVectors(self.data['bi_uvcoord2'], self.fov, 
                                       scale, self.pulse_ft),
                      utils.ftVectors(self.data['bi_uvcoord3'], self.fov, 
                                       scale, self.pulse_ft))
            
            for beta in self.betas:
                
                # (a) solve for Z while keeping x constant
                Z = utils.mostLikelyPatches(x, beta, self.data, 
                                            self.patch_size, self.gmm)
                
                # (b) solve for x while keeping Z constant
                x = utils.taylorExpansion(x, Z, beta, self.data, gammas,
                                          self.patch_size, lam=lam)
                
                if display:
                    self.__axs[0].clear()
                    self.__axs[1].clear()
                    self.__display(x, Z, 'Scale: ' + str(scale) + '\n' +\
                                          r'$\beta$: ' + str(beta))
            self.history[scale] = x
        
        if display:
            plt.ioff()
    
        self.res = np.rot90(utils.upscaleImage(x, self.fov, 
                                               self.naxis, self.pulse),2)
        return self.res
    
    def score(self, ref_im):
        
        """
        Align the reconstruction using phase correlation and calculate the MSE 
        and SSIM.
        
        Inputs:
        - ref_im: A numpy array of shape (N,N) contaning a reference image.                  
        Outputs:
        - mse: Mean squared error (float).
        - ssim: Structural similarity index (float).
        """
        
        if isinstance(self.res, type(None)):
            raise Exception('Result is not yet aviable.')
            
        shift = register_translation(ref_im, self.res)[0]
        shifted_res = fourier_shift(np.fft.fft2(self.res), shift)
        shifted_res = np.real(np.fft.ifft2(shifted_res))
        
        mse = np.linalg.norm(shifted_res - ref_im)
        drange = np.max(shifted_res) - np.min(shifted_res)
        ssim = compare_ssim(ref_im, shifted_res, data_range=drange)
        
        return mse, ssim
    
    def saveFits(self, filename):
        
        """
        Save the reconstruction in FITS.
        
        Inputs:
        - filename: Name for the file to be created (String).
        """
        
        if isinstance(self.res, type(None)):
            raise Exception('Result is not yet aviable.')
        
        header = fits.Header()
        header['NAXIS1'] = self.naxis
        header['NAXIS2'] = self.naxis
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CDELT1'] = - self.fov/(np.pi/180 * self.naxis)
        header['CDELT2'] = self.fov/(np.pi/180 * self.naxis)
        header['BUNIT'] = 'JY/PIXEL'
        
        hdu = fits.PrimaryHDU(self.res, header=header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        
        print("Saved as '%s'." %(filename))
    
    def showResult(self, ref_im=None):
        
        """
        Show the reconstructed image (in comparison to the source image).
        
        Inputs:
        - ref_im: A numpy array of shape (N,N) contaning a reference image. 
                  Default is None.   
        """
        
        if isinstance(self.res, type(None)):
            raise Exception('Result is not yet aviable.')        
        
        fov_mas = self.fov * 1e6 * 3600 * 180 / np.pi
        ticks = np.linspace(0, self.naxis-1, 7)
        ticklabels = np.linspace(fov_mas/2, -fov_mas/2, 7, dtype=int)
        
        if not isinstance(ref_im, type(None)):
            
            fig, axs = plt.subplots(1, 2)
            plt.subplots_adjust(wspace=-0.1)
            
            
            minVal = np.min([ref_im,self.res])
            maxVal = np.max([ref_im,self.res])
            
            im = axs[0].imshow(ref_im, cmap='gray', vmin=minVal, vmax=maxVal)
            temp = fig.colorbar(im, ax=axs[0], shrink=0.575, label='Jy/pixel')
            temp.remove()
            axs[0].set_xticks(ticks)
            axs[0].set_xticklabels(ticklabels)
            axs[0].set_yticks(ticks)
            axs[0].set_yticklabels(ticklabels)
            axs[0].set_title('Reference') 
            axs[0].set_xlabel('Right Ascension [$\mu$as]')
            axs[0].set_ylabel('Declination [$\mu$as]')

            im = axs[1].imshow(self.res, cmap='gray', vmin=minVal, vmax=maxVal)
            axs[1].set_title('Reconstruction')
            axs[1].set_xticks(ticks)
            axs[1].set_xticklabels(ticklabels)
            axs[1].set_yticks([])
            axs[1].set_xlabel('Right Ascension [$\mu$as]')
            fig.colorbar(im, ax=axs[1], shrink=0.575, label='Jy/pixel')
            
            plt.show()
    
        else:
            
            fig, ax = plt.subplots(1, 1)
            
            im = plt.imshow(self.res, cmap='gray')
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_title('Reconstruction') 
            ax.set_xlabel('Right Ascension [$\mu$as]')
            ax.set_ylabel('Declination [$\mu$as]')
            fig.colorbar(im, ax=ax, label='Jy/pixel')
            
            plt.show()
    
    def __display(self, x, Z, title):
        
        """
        Auxiliary method for displaying partial solutions.
        """

        minVal = np.min([x,Z])
        maxVal = np.max([x,Z])
        
        scale = x.shape[0]
        fov_muas = self.fov * 1e6 * 3600 * 180 / np.pi
        ticks = np.linspace(0, scale-1, 7)
        ticklabels = np.linspace(fov_muas/2, -fov_muas/2, 7, dtype=int)
        
        self.__fig.suptitle(title)
        im = self.__axs[0].imshow(np.rot90(Z,2), cmap='gray', 
                                  vmin=minVal, vmax=maxVal)
        temp = self.__fig.colorbar(im, ax=self.__axs[0], shrink=0.575, 
                                   label='Jy/pixel')
        temp.remove()
        self.__axs[0].set_xticks(ticks)
        self.__axs[0].set_xticklabels(ticklabels)
        self.__axs[0].set_yticks(ticks)
        self.__axs[0].set_yticklabels(ticklabels)
        self.__axs[0].set_title('Combined \n Patch Priors') 
        self.__axs[0].set_xlabel('Right Ascension [$\mu$as]')
        self.__axs[0].set_ylabel('Declination [$\mu$as]')

        im = self.__axs[1].imshow(np.rot90(x,2), cmap='gray', 
                                  vmin=minVal, vmax=maxVal)
        self.__axs[1].set_title('Image Coefficients')
        self.__axs[1].set_xticks(ticks)
        self.__axs[1].set_xticklabels(ticklabels)
        self.__axs[1].set_yticks([])
        self.__axs[1].set_xlabel('Right Ascension [$\mu$as]')
        
        if(self.__cbar):
            self.__cbar.remove()
        self.__cbar = self.__fig.colorbar(im, ax=self.__axs[1], shrink=0.575, 
                                          label='Jy/pixel')
            
        self.__fig.canvas.draw()
        plt.pause(0.001)