import numpy as np
import pywt
from scipy.ndimage import shift
from skimage.transform import rotate
from scipy.special import expit
import numpy as np
import pywt
from skimage.transform import rotate, resize
from scipy.ndimage import shift
import plotly.express as px
import copy  

def scale(x, min_val=0, max_val=1):
    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * (max_val - min_val) / 2
    else:
        x = (x - x.min()) / (x.max() - x.min())
        x = x * (max_val - min_val) + min_val
    return x

def shift_rotate_img(img, shift_x, shift_y, angle, reverse=False):
    if reverse:
        shift_x = -shift_x
        shift_y = -shift_y
        angle = -angle
    img = shift(img, (shift_y, shift_x), mode='nearest')
    img = rotate(img, angle, resize=False, mode='symmetric')
    return img

def coeff_to_img(coeff, size, order=0):
    wavelet_img = resize(np.abs(coeff), size, order=0)
    wavelet_img = scale(wavelet_img, 0, 1)
    return wavelet_img

def compute_dwt(img, wavelet):
    coeffs = pywt.wavedec2(img, wavelet=wavelet)
    wavelet_img, _ = pywt.coeffs_to_array(coeffs)

    normalized_coeffs = copy.deepcopy(coeffs)
    normalized_coeffs[0] = scale(normalized_coeffs[0], 0, 1)
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cH = scale(cH, 0, 1)
        cV = scale(cV, 0, 1)
        cD = scale(cD, 0, 1)
        normalized_coeffs[i] = (cH, cV, cD)
    norm_wavelet_img, _ = pywt.coeffs_to_array(normalized_coeffs)
   


    return wavelet_img, norm_wavelet_img, coeffs


def plot_histogram(coeff, title):
    fig = px.histogram(
        x=coeff.flatten(),
        nbins=32,
        title=title,
        labels={'x': 'Value', 'y': 'Count'},
        histnorm='percent',
        marginal='box',
        range_y=[0, 100],
    )
    fig.update_layout(bargap=0.5)
    return fig

def reconstruct_from_coeffs(coeffs, selected_levels, wavelet):
    coeffs = copy.deepcopy(coeffs)
    if 0 not in selected_levels:
        coeffs[0] = np.zeros_like(coeffs[0])
    for i in range(1, len(coeffs)):
        if i not in selected_levels:
            coeffs[i] = (
                np.zeros_like(coeffs[i][0]), 
                np.zeros_like(coeffs[i][1]), 
                np.zeros_like(coeffs[i][2])
            )
    rec_img = pywt.waverec2(coeffs, wavelet=wavelet)
    return rec_img

def quantize_dwt(coeffs, bits_per_level=None):
    """
    Uniformly quantize wavelet coefficients level-by-level, using bits_per_level
    dict {level_idx: num_bits}, e.g., {1: 4, 2: 2, ...}
    """
    coeffs_q = copy.deepcopy(coeffs)
    
    if bits_per_level is None:
        bits_per_level = {}
    
    # Quantize each detail level
    for level_idx in range(1, len(coeffs_q)):
        cH, cV, cD = coeffs_q[level_idx]
        bits = bits_per_level.get(level_idx, 8)
        
        def quantize_subband(band):
            max_abs = np.max(np.abs(band))
            if np.isclose(max_abs, 0):
                return band
            if bits == 1:
                # For 1-bit quantization, use the sign of the coefficient.
                return np.where(band < 0, -max_abs, max_abs)
            N = 2**bits  
            step = 2 * max_abs / (N - 1)
            band_q = np.round(band / step) * step
            band_q = np.clip(band_q, -max_abs, max_abs)
            return band_q
        
        coeffs_q[level_idx] = (quantize_subband(cH),
                                quantize_subband(cV),
                                quantize_subband(cD))
    
    return coeffs_q




    



