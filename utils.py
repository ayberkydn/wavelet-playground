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

def normalize(x):
    """Linearly scales array x to [0..1] unless x is constant."""
    if np.allclose(x.max(), x.min()):
        return x
    else:
        x = (x - x.min()) / (x.max() - x.min())
    return x

def shift_rotate_img(img, shift_x, shift_y, angle):
    """Shifts and rotates an image."""
    img = shift(img, (shift_y, shift_x), mode='nearest')
    img = rotate(img, angle, resize=False, mode='symmetric')
    return img

def coeff_to_img(coeff, size, order=0):
    wavelet_img = np.abs(np.clip(coeff, -1, 1))
    wavelet_img = resize(wavelet_img, size, order=0)
    return wavelet_img

def compute_dwt(img, wavelet):
    coeffs = pywt.wavedec2(img, wavelet=wavelet)
    # coeffs = pywt.wavedec2(img, wavelet=wavelet, level=levels)
    wavelet_img, _ = pywt.coeffs_to_array(coeffs)
    wavelet_img = np.abs(np.clip(wavelet_img, -1, 1))

    return wavelet_img, coeffs

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
        nbins=255,
        title=title,
        labels={'x': 'Coefficient Value', 'y': 'Frequency'}
    )
    fig.update_layout(bargap=0.1)
    return fig






    



