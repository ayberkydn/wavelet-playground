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

def scale(x, min_val, max_val):
    if np.allclose(x.max(), x.min()):
        return x
    else:
        x = (x - x.min()) / (x.max() - x.min())
        x = x * (max_val - min_val) + min_val
    return x

def shift_rotate_img(img, shift_x, shift_y, angle):
    img = shift(img, (shift_y, shift_x), mode='nearest')
    img = rotate(img, angle, resize=False, mode='symmetric')
    return img

def coeff_to_img(coeff, size, order=0):
    wavelet_img = np.abs(np.clip(coeff, -1, 1))
    wavelet_img = resize(wavelet_img, size, order=0)
    return wavelet_img

def compute_dwt(img, wavelet):
    coeffs = pywt.wavedec2(img, wavelet=wavelet)
    wavelet_img, _ = pywt.coeffs_to_array(coeffs)
    # cAs = []
    # for i in range(1, len(coeffs)):
    #     removed_levels = [n for n in range(len(coeffs)) if n >= i]
    #     cAs.append(reconstruct_dwt(coeffs, removed_levels, wavelet))

    cHs = [coeff[0] for coeff in coeffs[1:]]
    cVs = [coeff[1] for coeff in coeffs[1:]]
    cDs = [coeff[2] for coeff in coeffs[1:]]

    return wavelet_img, coeffs, cHs, cVs, cDs

def reconstruct_dwt(coeffs, wavelet, percent_thresholds_per_level=None, val_thresholds_per_level=None):
    if percent_thresholds_per_level is not None:
        assert val_thresholds_per_level is None, "Cannot use both percent and value thresholds"
    elif val_thresholds_per_level is not None:
        assert percent_thresholds_per_level is None, "Cannot use both percent and value thresholds"
    else:
        raise ValueError("Either percent_thresholds_per_level or val_thresholds_per_level must be provided")
    
    coeffs = copy.deepcopy(coeffs)

    # Handle percent-based thresholding (unchanged)
    if percent_thresholds_per_level is None:
        percent_thresholds_per_level = {}
    for level_idx in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level_idx]
        percent = percent_thresholds_per_level.get(level_idx, 0)
        if 0 < percent < 100:
            combined = np.abs(np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()]))
            cutoff = np.percentile(combined, percent)
            cH = np.where(np.abs(cH) < cutoff, 0, cH)
            cV = np.where(np.abs(cV) < cutoff, 0, cV)
            cD = np.where(np.abs(cD) < cutoff, 0, cD)
        elif percent == 100:
            cH = np.zeros_like(cH)
            cV = np.zeros_like(cV)
            cD = np.zeros_like(cD)
        coeffs[level_idx] = (cH, cV, cD)

    # --------------- NEW CODE: magnitude (value) thresholding ---------------
    if val_thresholds_per_level is not None:
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            threshold_val = val_thresholds_per_level.get(level_idx, 0.0)
            if threshold_val > 0:
                cH = np.where(np.abs(cH) < threshold_val, 0, cH)
                cV = np.where(np.abs(cV) < threshold_val, 0, cV)
                cD = np.where(np.abs(cD) < threshold_val, 0, cD)
            coeffs[level_idx] = (cH, cV, cD)
    # -------------------------------------------------------------------------

    return pywt.waverec2(coeffs, wavelet=wavelet)

def plot_histogram(coeff, title):
    """Plots a histogram of the given coefficients."""
    fig = px.histogram(
        x=coeff.flatten(),
        nbins=64,
        title=title,
        labels={'x': 'Coefficient Value', 'y': 'Frequency'}
    )
    fig.update_layout(bargap=0.1)
    return fig






    



