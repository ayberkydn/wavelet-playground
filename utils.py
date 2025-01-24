import numpy as np
import pywt
from scipy.ndimage import shift
from skimage.transform import rotate
import imageio
import tempfile
import numpy as np
import pywt
from skimage.transform import rotate, resize
from scipy.ndimage import shift


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

def compute_dwt(img, levels, wavelet='haar'):
    """
    Computes a 2D wavelet transform up to 'levels'.
    Returns:
      - wavelet_img      (array): visualization of wavelet details (approx set to 0)
      - coeffs           (list) : raw wavelet coefficients for each level
    """
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=levels)

    # Zero out approximation to highlight details for wavelet_img
    coeffs_for_visualization = coeffs.copy()
    coeffs_for_visualization[0] = np.zeros_like(coeffs_for_visualization[0])
    wavelet_img, _ = pywt.coeffs_to_array(coeffs_for_visualization)

    return normalize(wavelet_img), coeffs

def create_wavelet_coefficients_video(coeffs, fps=3, target_size=(512, 512)):
    """
    Creates an MP4 video of all wavelet coefficients (LL, LH, HL, HH) at each wavelet level.
    Returns raw MP4 bytes that can be passed to st.video(...).
    
    coeffs format: [cA_n, (cH_1, cV_1, cD_1), ..., (cH_n, cV_n, cD_n)]
    """
    frames = []

    for level_index in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level_index]  # horizontal, vertical, diagonal details
        
        cH = resize(cH + 128, target_size, order=0)
        cV = resize(cV + 128, target_size, order=0)
        cD = resize(cD + 128, target_size, order=0)
        # Combine coefficients into a single image
        combined_frame = np.hstack((cH, cV, cD))

        frames.append(combined_frame)

    if not frames:
        return None

    # Write MP4 to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
        with imageio.get_writer(tmp_file.name, format='ffmpeg', fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        tmp_file.seek(0)
        video_bytes = tmp_file.read()

    return video_bytes
