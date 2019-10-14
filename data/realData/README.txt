

** PLEASE READ THIS FULLY BEFORE WORKING WITH THIS DATA **


Here we provide a sample of data from the Boston University (BU) Blazar Group (http://www.bu.edu/blazars/VLBAproject.html) prepared to have the same format that we have used throughout this dataset and website. Although the wavelength of this data is large enough that some phase calibration can (and has) be done, this data provides a means to test algorithms on a large number of real measurements. Since this is real data 


** THERE ARE NO GROUND TRUTH IMAGES ** 


associated with the data. However, for reference we provide the images and CLEAN models produced by the BU Blazar Group in 'origBUData'. The MOD files contain the CLEAN models produced by the BU Group. To view and manipulate these models you can load the data into a standard package such as CASA. The IMAP files contain the FITS images produced when you convolve the CLEAN model with a Gaussian Blur Kernel. The UVP files contain the original UVFITS data provided on BU's website. When comparing images keep in mind that these 


** REFERENCE IMAGES ARE UP-DOWN FLIPPED RELATIVE TO THE OTHER IMAGES IN THE DATASET **


Please look the PNG images in 'targetImgs*' to view the correct flipping of the IMAP files in 'origBUData'. The FOV of each of the images can be determined by inspecting the header information of the IMAP files (you can view this by opening the file in a standard text editor). 


** TO CALCULATE THE FIELD OF VIEW (FOV) FOR EACH IMAGE ** 


Multiply the value of NAXIS by the value of CDELT to obtain the FOV in degrees. To convert from degrees to arcseconds multiply by 3600. Specifically, FOV = NAXIS*CDELT*3600 arcseconds. Additionally, the


** ABSOLUTE TIME MEASUREMENTS PROVIDED WITH THIS DATA ARE INCORRECT **


However, we still include a time value to indicate which measurements were taken at the same time. 