
import numpy as np
import pywt
from scipy.ndimage import shift


def normalize(x):
    if np.allclose(x.max(), x.min()):
        return x
    else:
        x = (x - x.min()) / (x.max() - x.min())
    return x

def normalize_coeffs(coeffs):    
    '''
      coeffs format: [cA_n, (cH_1, cV_1, cD_1), ..., (cH_n, cV_n, cD_n)]
    '''
    ll = coeffs[0]
    norm_coeffs = [normalize(ll)] + [
        (normalize(cH), normalize(cV), normalize(cD)) 
        for cH, cV, cD in coeffs[1:]
    ]
    return norm_coeffs

def shift_img(img, shift_x, shift_y):
    return shift(img, (shift_y, shift_x), mode='nearest')

def compute_dwt(img, levels, wavelet='haar'):
    # Subtract mean to have zero-mean image (if desired)
    img = img - img.mean()

    # Ensure we don't exceed the maximum possible decomposition level
    max_level = int(np.floor(np.log2(min(img.shape))))
    levels = min(levels, max_level) - 1
    
    # Decompose using the selected wavelet and level
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=levels)
    
    # Convert wavelet coefficients into a single image array
    dwt_img, _ = pywt.coeffs_to_array(coeffs)
    # Create a normalized version of that same decomposition
    norm_dwt_img, _ = pywt.coeffs_to_array(normalize_coeffs(coeffs))

    return normalize(dwt_img), normalize(norm_dwt_img)
