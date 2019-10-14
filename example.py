import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import oifits as oi
import vlbi

# flux: 1.0
oidata = oi.open('./data/syntheticData/staticNatural/UVdata/' +\
                 'natural-03-10.oifits')
fitsdata = fits.open('./data/syntheticData/staticNatural/' +\
                     'targetImgs/natural-03-10.fits')[0]

src_im = fitsdata.data
target_im = plt.imread('./data/syntheticData/targets/natural-03.pgm')[:,:,0]
target_im = target_im.astype(np.float64)
target_im /= np.max(target_im)
target_im *= np.max(src_im)

fov = np.abs(fitsdata.header['CDELT1']) * np.pi/180 * fitsdata.header['NAXIS1']
print('fov:      %0.2f mas \n      %0.4e rad'
      %(fov * 1e6 * 3600 * 180 / np.pi, fov))
naxis = fitsdata.header['NAXIS1']

chirp = vlbi.CHIRP(oidata, fov, naxis)
res = chirp.reconstruct(display=True)

chirp.showResult(ref_im=src_im)
mse, ssim = chirp.score(target_im)
print('MSE:  %0.6f \nSSIM: %0.6f' %(mse, ssim))