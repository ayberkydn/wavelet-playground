import numpy as np
import pywt
from scipy.ndimage import shift
from skimage.transform import rotate, resize
from scipy.special import expit
import numpy as np
import pywt
import plotly.express as px
import copy  
import streamlit as st

def scale(x, min_val=0, max_val=1):
    if x.ndim == 3:
        for i in range(x.shape[2]):
            x[..., i] = scale(x[..., i], min_val, max_val)
        return x
    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * (max_val - min_val) / 2
    else:
        x = (x - x.min()) / (x.max() - x.min())
        x = x * (max_val - min_val) + min_val
    return x

def transform_img(img, shift_x, shift_y, angle, scale_factor=1.0, reverse=False):
    """Apply shift, rotation, and scaling to image."""
    if reverse:
        shift_x = -shift_x
        shift_y = -shift_y
        angle = -angle
        scale_factor = 1/scale_factor if scale_factor != 0 else 1.0

    if img.ndim == 3:
        for i in range(img.shape[2]):
            img[..., i] = transform_img(img[..., i], shift_x, shift_y, angle, scale_factor, reverse)
        return img

    # Apply shift and rotation
    img = shift(img, (shift_y, shift_x), mode='reflect')
    img = rotate(img, angle, resize=False, mode='reflect')

    return img

def coeff_to_img(coeff, size, order=0):
    wavelet_img = resize(np.abs(coeff), size, order=0)
    wavelet_img = scale(wavelet_img, 0, 1)
    return wavelet_img

def compute_dwt(img, wavelet):
    if img.ndim == 3:
        coeffs = []
        wavelet_imgs = []
        norm_wavelet_imgs = []
        for i in range(img.shape[2]):
            wavelet_img, norm_wavelet_img, coeff = compute_dwt(img[..., i], wavelet)
            wavelet_imgs.append(wavelet_img)
            norm_wavelet_imgs.append(norm_wavelet_img)
            coeffs.append(coeff)
        return np.stack(wavelet_imgs, axis=-1), np.stack(norm_wavelet_imgs, axis=-1), coeffs

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

def plot_histogram(coeff, title, use_abs=False):
    if use_abs:
        x = np.abs(coeff.flatten())
    else:
        x = coeff.flatten()

    #remove small values
    x = x[np.abs(x) > 1e-3]
    
    fig = px.histogram(
        x=x,
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
    if isinstance(coeffs[0], list):
        rec_imgs = []
        for i in range(len(coeffs)):
            rec_img = reconstruct_from_coeffs(coeffs[i], selected_levels, wavelet)
            rec_imgs.append(rec_img)
        return np.stack(rec_imgs, axis=-1)

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
    if isinstance(coeffs[0], list):
        quantized_coeffs = []
        for i in range(len(coeffs)):
            quantized_coeff = quantize_dwt(coeffs[i], bits_per_level)
            quantized_coeffs.append(quantized_coeff)
        return quantized_coeffs

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
            N = 2**bits  
            step = 2 * max_abs / (N - 1)
            band_q = np.round(band / step) * step
            band_q = np.clip(band_q, -max_abs, max_abs)
            return band_q
        
        coeffs_q[level_idx] = (quantize_subband(cH),
                                quantize_subband(cV),
                                quantize_subband(cD))
    
    return coeffs_q

def plot_image(img, title="", colorscale="gray"):
    """Convert image array to plotly figure with proper aspect ratio."""
    fig = px.imshow(
        img,
        title=title,
        color_continuous_scale=colorscale,
        aspect='equal'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def threshold_dwt(coeffs, threshold):
    if isinstance(coeffs[0], list):
        thresholded_coeffs = []
        for i in range(len(coeffs)):
            thresholded_coeff = threshold_dwt(coeffs[i], threshold)
            thresholded_coeffs.append(thresholded_coeff)
        return thresholded_coeffs

    coeffs = copy.deepcopy(coeffs)
    coeffs[0] = coeffs[0] * (np.abs(coeffs[0]) > threshold)
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cH = cH * (np.abs(cH) > threshold)
        cV = cV * (np.abs(cV) > threshold)
        cD = cD * (np.abs(cD) > threshold)
        coeffs[i] = (cH, cV, cD)
    return coeffs







