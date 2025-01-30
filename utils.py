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
import streamlit as st

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

def reconstruct_dwt(coeffs, removed_levels, wavelet):
    for level in removed_levels:
        coeffs[level] = (
            np.zeros_like(coeffs[level][0]),
            np.zeros_like(coeffs[level][1]),
            np.zeros_like(coeffs[level][2])
        )
    st.write(len(coeffs))
    return pywt.waverec2(coeffs, wavelet=wavelet)

def plot_histogram(coeff, title):
    """Plots a histogram of the given coefficients."""
    fig = px.histogram(
        x=coeff.flatten(),
        nbins=16,
        title=title,
        labels={'x': 'Coefficient Value', 'y': 'Frequency'}
    )
    fig.update_layout(bargap=0.1)
    return fig






    



