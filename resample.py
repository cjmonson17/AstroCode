
# coding: utf-8

# # RESAMPLING PHOENIX FILES TO MATCH IGRINS

# ## Table of Contents <a class ="tocSkip">
# line 16-    Spectra-Resampling-Code
# line        imports
#             Importing spectra files
#
#
#
#






#-----------------------Files Import------------
from __future__ import print_function, division, absolute_import

import numpy as np


# ---------------DEFINING SPECTRE RESAMPLING

def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        old_spec_wavs, last dimension must correspond to the shape of
        old_spec_wavs. Extra dimensions before this may be used to
        include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    Returns
    -------
    res_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_spec_wavs, other dimensions are the same as
        spec_fluxes.
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in
        res_fluxes. Only returned if spec_errs was specified.
    """

    # Arrays of left-hand sides and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0]
    spec_lhs[0] -= (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0]
    filter_lhs[0] -= (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1]
    filter_lhs[-1] += (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]


    if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
        raise ValueError("spectres: The new wavelengths specified must fall"
                         " within the range of the old wavelength values.")

    # Generate output arrays to be populated
    res_fluxes = np.zeros(spec_fluxes[..., 0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape"
                             " as spec_fluxes.")
        else:
            res_fluxerrs = np.copy(res_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, loop over new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # If new bin is fully within one old bin these are the same
        if stop == start:

            res_fluxes[..., j] = spec_fluxes[..., start]
            if spec_errs is not None:
                res_fluxerrs[..., j] = spec_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:

            start_factor = ((spec_lhs[start+1] - filter_lhs[j])
                            / (spec_lhs[start+1] - spec_lhs[start]))

            end_factor = ((filter_lhs[j+1] - spec_lhs[stop])
                          / (spec_lhs[stop+1] - spec_lhs[stop]))

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate res_fluxes spectrum and uncertainty arrays
            f_widths = spec_widths[start:stop+1]*spec_fluxes[..., start:stop+1]
            res_fluxes[..., j] = np.sum(f_widths, axis=-1)
            res_fluxes[..., j] /= np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                e_wid = spec_widths[start:stop+1]*spec_errs[..., start:stop+1]

                res_fluxerrs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                res_fluxerrs[..., j] /= np.sum(spec_widths[start:stop+1])

            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor

    # If errors were supplied return the res_fluxes spectrum and error arrays
    if spec_errs is not None:
        return res_fluxes, res_fluxerrs

    # Otherwise just return the res_fluxes spectrum array
    else:
        return res_fluxes



# import spectreResample
# [Resampling Code](http://localhost:8888/notebooks/CJ_Astro/resample.ipynb#Spectra-Resampling-Code:-courtesy-of-ACCarnall)

# ### Importing Python packages

# In[1]:

# ---------------RESAMPLE CODE

# ------IMPORTS




#--------------------------Package Import---------------------------


from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import scipy as sp
import numpy as np
from scipy import *
from decimal import Decimal
import numpy as np #imports numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#import seaborn as sns # provides a high-level interface to draw statitistical graphics
from pylab import *
#%config InlineBackend.figure_format = 'retina' #makes images clearer
#%matplotlib inline
#^put plots in line?
from scipy.constants import parsec as pc
#from Practice1 import sig_out110

# from PHX_spec import StelFluxEarth03800_15, StelFluxEarth04800_15


#--------------------------------



# --------------------------Important values
dM67_PC = 900 # Distance to M67 in parsecs
pc_cm = 3.0856776e18
#Spc_Scm = 9.5214e36   # 1 sq pc = 9.5214 * 10^36  sq cm
dM67_cm = dM67_PC * pc_cm # Distance to M67 in centimeters
InvSq = 4*pi*dM67_cm**2
cm_A = 1e8 # converting centimeters to angstroms
Angstroms = 1e4 # microns
Ang_m  = 1e-10 # meters



#----------------IGRINS Import--------------

# In[5]:


hdu = fits.open("../Desktop/C_A_Files/SDCH_20150425_0064.spec_a0v.fits")
#hdu.info()
var = fits.open("../Desktop/C_A_Files/SDCH_20150425_0064.variance.fits")
#var.info()


# #### Calling certain files and renaming them

dat = hdu[0]
wav = hdu[1]
varx = var[0]



#--------------PHX WAVELENGTH Import---------------
# ## Importing PHOENIX Wave Files

varPHX = fits.open("../Desktop/C_A_Files/WAVE_PHOENIX-ACES-AGSS-COND-2011 (2).fits")
PHXwave = varPHX[0].data
varPHX.info()
trimA = 1108332
trimB = 1174999


#---------------PHX Flux Imports
# # Importing Spectra and Headers from PHOENIX
# ### Same exact thing as in TableFits2.1

hdu03800_15 = fits.open("../Desktop/C_A_Files/lte03800-1.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux03800_15 = hdu03800_15[0].data #in terms of joules
hdu03800_15_file = get_pkg_data_filename("../Desktop/C_A_Files/lte03800-1.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

PHXREFF03800_15 = fits.getval(hdu03800_15_file, 'PHXREFF')
StelArea03800_15 = 4*pi*PHXREFF03800_15**2 # cm^2     # surface area in square centimeters
StelFluxEarth03800_15 = flux03800_15/cm_A*StelArea03800_15/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)

hdu03800_25 = fits.open("../Desktop/C_A_Files/lte03800-2.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux03800_25 = hdu03800_25[0].data #in terms of joules
hdu03800_25_file = get_pkg_data_filename("../Desktop/C_A_Files/lte03800-2.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

PHXREFF03800_25 = fits.getval(hdu03800_25_file, 'PHXREFF')
StelArea03800_25 = 4*pi*PHXREFF03800_25**2 # cm^2     # surface area in square centimeters
StelFluxEarth03800_25 = flux03800_25/cm_A*StelArea03800_25/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)


hdu04800_15 = fits.open("../Desktop/C_A_Files/lte04800-1.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux04800_15 = hdu04800_15[0].data #in terms of joules
hdu04800_15_file = get_pkg_data_filename("../Desktop/C_A_Files/lte04800-1.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

PHXREFF04800_15 = fits.getval(hdu03800_15_file, 'PHXREFF')
StelArea04800_15 = 4*pi*PHXREFF04800_15**2 # cm^2     # surface area in square centimeters
StelFluxEarth04800_15 = flux04800_15/cm_A*StelArea04800_15/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)

hdu04800_25 = fits.open("../Desktop/C_A_Files/lte04800-2.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux04800_25 = hdu04800_25[0].data #in terms of joules
hdu04800_25_file = get_pkg_data_filename("../Desktop/C_A_Files/lte04800-2.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

PHXREFF04800_25 = fits.getval(hdu03800_25_file, 'PHXREFF')
StelArea04800_25 = 4*pi*PHXREFF04800_25**2 # cm^2     # surface area in square centimeters
StelFluxEarth04800_25 = flux04800_25/cm_A*StelArea04800_25/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)


# --------------------------------4000 K_40

# In[13]:



hdu04000_40 = fits.open("../Desktop/C_A_Files/lte04000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux04000_40 = hdu04000_40[0].data #in terms of joules
hdu04000_40_file = get_pkg_data_filename("../Desktop/C_A_Files/lte04000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")


PHXREFF04000_40 = fits.getval(hdu04000_40_file, 'PHXREFF')


StelArea04000_40 = 4*pi*PHXREFF04000_40**2 # cm^2     # surface area in square centimeters
StelFluxEarth04000_40 = flux04000_40/cm_A*StelArea04000_40/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)
#print(len(StelFluxEarth04000_40))
#print(len(PHXwave))



# ## ---------------------------------5100 K_40

# In[14]:


hdu05100_40 = fits.open("../Desktop/C_A_Files//lte05100-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")  # open a FITS file
flux05100_40 = hdu05100_40[0].data #in terms of joules
hdu05100_40_file = get_pkg_data_filename("../Desktop/C_A_Files/lte05100-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")


PHXREFF05100_40 = fits.getval(hdu05100_40_file, 'PHXREFF')

StelArea05100_40 = 4*pi*PHXREFF05100_40**2 # cm^2     # surface area in square centimeters
StelFluxEarth05100_40 = flux05100_40/cm_A*StelArea05100_40/(InvSq) # cm^2/pc     # conversion for stellar flux as seen from Earth (multiply by flux######)



def Luminosity(flux): # flux at earth and radius = distance Earth
    lum = flux * InvSq * Ang_m
    return lum
print(Luminosity(StelFluxEarth03800_15))


# ### Important Values and conversions

# In[7]:





# ---------------------------### Order 110 editing

# In[8]:


o110 = 12

flux110 = dat.data[o110]
fluxcut110 = flux110[~np.isnan(flux110)]

wav110 = wav.data[o110] * Angstroms
wavecut110 = wav110[~np.isnan(flux110)]

scale_factor110 = hdu[3].data[o110]/hdu[4].data[o110]
sig_raw110 = np.sqrt(var[0].data[o110])
sig_scaled110 = sig_raw110/scale_factor110
sig110 = sig_scaled110
sigcut110 = sig110[~np.isnan(flux110)]
sg_110 = sigcut110

# ----------------------- Trimming out saturated data

# In[9]:

trim110A = 16230
trim110B = 16400

trim110 = (wavecut110 > trim110A ) & (wavecut110 < trim110B)
trim110.sum()



# In[10]: ------------------Renaming trimmed values


wavetrim110 = wavecut110[trim110]
fluxtrim110 = fluxcut110[trim110]
sigtrim110 = sigcut110[trim110]
#print(len(wavetrim110))
#print(len(fluxtrim110))
#print(len(sigtrim110))
#print(wavetrim110)
#print(fluxtrim110)
#print(sigtrim110)

# -------------------Resampling  (see bottom of page for code)

# In[17]: Resamples the PHX spectra to fit with IGRINS ORDER 110
reSFE04000_40 = spectres(wavetrim110,PHXwave,StelFluxEarth04000_40)
reSFE05100_40 = spectres(wavetrim110,PHXwave,StelFluxEarth05100_40)
reSFE03800_15 = spectres(wavetrim110,PHXwave,StelFluxEarth03800_15)
reSFE04800_15 = spectres(wavetrim110,PHXwave,StelFluxEarth04800_15)
reSFE03800_25 = spectres(wavetrim110,PHXwave,StelFluxEarth03800_25)
reSFE04800_25 = spectres(wavetrim110,PHXwave,StelFluxEarth04800_25)

# --------------------Scaling the values from IGRINS


med_trimflux110 = np.median(fluxtrim110)
fluxtrim110_med = fluxtrim110/med_trimflux110

med_trimsig110 = np.median(sigtrim110)
sigtrim110_med  =  sigtrim110/med_trimsig110

# ----------------- Scaling the values from PHOENIX


med_reSFE05100_40 = np.median(reSFE05100_40)
reSFE05100_40_med = reSFE05100_40/med_reSFE05100_40

med_reSFE04000_40 = np.median(reSFE04000_40)
reSFE04000_40_med = reSFE04000_40/med_reSFE04000_40
#print(reSFE05100_40)

med_reSFE03800_15 = np.median(reSFE03800_15)
reSFE03800_15_med = reSFE03800_15/med_reSFE03800_15

med_reSFE04800_15 = np.median(reSFE04800_15)
reSFE04800_15_med = reSFE04800_15/med_reSFE04800_15

med_reSFE03800_25 = np.median(reSFE03800_25)
reSFE03800_25_med = reSFE03800_25/med_reSFE03800_25

med_reSFE04800_25 = np.median(reSFE04800_25)
reSFE04800_25_med = reSFE04800_25/med_reSFE04800_25


# ------------------------Doppler Shift

# #### Speed of light conversions

c = 299792458.0 # meters/sec
c_m = c # meters/sec
c_cm = c*1e2 #centimeters/sec
c_km = c*1e-3 #kilometers/sec


# In[22]: ------------------ Actual Doppler Shift Equations


v1 = 30
#v2 = 20
#c = 3e10 # cm/s
v2 = 40 #cm/s [redshift]
#v2 = -.0001*c #cm/s [blueshift]
wave_obsR = (sqrt((1+v1/c_km)/(1-v1/c_km))) # [redshift]
wave_obsR2 = (sqrt((1+v2/c_km)/(1-v2/c_km))) # [redshift]

wave_obsB = (sqrt((1-v2/c_km)/(1+v2/c_km))) # [blueshift]
print("+++ Doppler shift")
print(16000 * (wave_obsR -1))
print(16000 * (wave_obsR2 -1))

#print(wave_obsR -1)
print(v1/19)
print(v2/19)
print("+++")


# In[23]: ------------Dopple shifting


wavetrim110_obsB = wavetrim110*wave_obsB # [blueshift]
wavetrim110_obsR = wavetrim110*wave_obsR # [redshift]
wavetrim110_obsR2 = wavetrim110*wave_obsR2 # [redshift]

#SFE05100trim = StelFluxEarth05100_40[trimA, trimB]
#PHXwavetrim_R = PHXwavetrim * wave_obsR
PHXerr = ""
PHXerr = .03 * reSFE05100_40_med

#---------------------File output


#------------------ ccdproc CCDDAta //Did not work very well

import numpy as np
from astropy.nddata import CCDData
import ccdproc
#ccd = CCDData(wavetrim110_obsR, unit='adu')
#ccd2 = CCDData(wavetrim110, unit='adu')
# ## Changing from array to .fits
#ccd3 = CCDData(PHXwave_R, unit='adu')
#ccd4 = CCDData(PHXwave, unit='adu')
# In[26]:


#dat.write('my_image.fits')
#reSFE04000_40.write('SFE04000.fits')
#ccd3.write('PHXwave_R.fits')
#ccd4.write('PHXwave.fits')




# ----------------------------- PLOTTING

# ------- With Doppler Shift


# ------plot Synthetic binary with redshift
'''
plt.plot(wavetrim110_obsR, reSFE04000_40_med, color = 'lightcoral', label = "5100 K-4.0")
# plot Synthetic binary with blueshift
#plt.plot(wavetrim110_obsB, reSFE05100_40_med, color = 'skyblue', label = "5100 K-4.0")

# Plot actual binary order-110
plt.plot(wavetrim110, fluxtrim110_med, color = 'gray', label = "Order 110 (5130 K)")
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel("Scaled flux")
#plt.ylabel("Flux [erg/s/$cm^2$/$\AA$]")
xlim(16320,16350)
#ylim(5e-16,2e-15)
plt.title("Synthetic Spectra")
plt.legend()
plt.grid()
plt.show()
plt.close()


# -------- Without Doppler Shift

plt.plot(wavetrim110, reSFE05100_40_med, color = 'gray', label = "5100 K-4.0")
plt.plot(wavetrim110, fluxtrim110_med, color = 'blue', label = "Order 110")
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel("Flux [erg/s/$cm^2$/$\AA$]")
#xlim(16170,16400)
#ylim(5e-16,2e-15)
plt.title("Synthetic Spectra")
plt.legend()
plt.grid()
plt.show()
plt.close()
'''

print("--------Index and Wavelength---------")
print(wavetrim110[850])
print(wavetrim110[980])
print(wavetrim110[1090])
print("-----------------")


#r = reSFE03800_25[30:960]
#o = reSFE03800_25[15:945]
#b = reSFE03800_25[0:930]
x = 850
y = 1090
V1i = int(v1/2)
V2i = int(v2/2)
A = x+V1i
B = y+V1i
#print(A)
#print(B)

r = reSFE03800_25[A:B]
o = reSFE03800_25[x:y]
b = reSFE04800_25[x-V2i:y-V2i]

r2 = reSFE03800_25[885:1125]

#plt.plot(wavetrim110_obsR, reSFE03800_25, color = 'firebrick', label = "3800 K-2.5")
#plt.plot(wavetrim110, reSFE03800_25, color = 'gray', label = "3800 K-2.5")
#plt.plot(wavetrim110_obsR2, reSFE03800_25, color = 'blue', label = "3800 K-2.5")

#plt.plot(wavetrim110_obsB, reSFE03800_25, color = 'lightcoral', label = "3800 K-2.5")


# Plot redshift
plt.plot(wavetrim110_obsR[x+V1i:y+V1i], r, color = 'lightcoral', label = "3800 K-2.5 (v=30)") #[880:1120] r + 1e-15
#plt.plot(wavetrim110_obsR2[885:1125], r2 + 1e-15, color = 'lightblue', label = "3800 K-2.5")



# Plot blueshift
plt.plot(wavetrim110_obsB[x-V2i:y-V2i], b, color = 'lightblue', label = "4800 K-2.5 (v=40)")


# Plot combination or redshift and blueshift
plt.plot(wavetrim110[x:y], (r + b), color = 'purple', label = "3800_2.5 + 4800_2.5")


plt.xlabel("Wavelength [$\AA$]")
plt.ylabel("Flux [erg/s/$cm^2$/$\AA$]")
#plt.annotate('local max', xy=(16330, .6e-12), xytext=(16340, .4e-12),
        #    arrowprops=dict(facecolor='black', shrink=0.05),
        #    )
#plt.ylabel("Flux [erg/s/$cm^2$/$\AA$]")
xlim(16325.5,16340)
#ylim(5e-16,2e-15)
plt.title("Synthetic spectra and Synthetic binary")
plt.legend()
plt.grid()

plt.savefig("SynBinFinal1.png")
plt.close()

from decimal import Decimal

m = 0
def LumDiff(b):
    some = 5
    length = len(b)
    m = 0
    for i in range(some):

        m = m + b[i]
        #print(b[i])
        #print(m)
    Avg = m/i
    return Avg
A = LumDiff(reSFE04800_25)
B = LumDiff(reSFE03800_25)
print("-------Difference in Wavelength------")
#print("reSFE04800_25: %f"%A)
#print("reSFE03800_25: %f"%B)
#print("reSFE04800_25 - reSFE03800_25: %f"%A-B)
print("-------------")
'''
c = Decimal(reSFE04800_25[500])
d = Decimal(reSFE03800_25[500])
print(c-d)
print(c+d)
a = Decimal(reSFE04800_25[1000])
b = Decimal(reSFE03800_25[1000])
print(a-b)
e = Decimal(reSFE04800_25[5])
f = Decimal(reSFE03800_25[5])
print(e-f)
g = Decimal(reSFE04800_25[1558])
h = Decimal(reSFE03800_25[1558])
print(g-h)
#print(len(reSFE03800_15))
#print(Decimal(StelFluxEarth03800_15[500]))
#print("Average Luminosity: %d"% (D))
'''


#return
# # Spectra Resampling Code: courtesy of ACCarnall
# [GitHub code](https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py)
