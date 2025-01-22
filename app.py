import streamlit as st
import numpy as np
import pywt
from skimage import data

# Set page config first (must be done before creating other Streamlit elements)
st.set_page_config(
    page_title="Shift + Wavelet Decomposition",
    layout="wide"
)

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
    levels = len(coeffs)
    norm_coeffs = [normalize(ll)] + [(normalize(cH), normalize(cV), normalize(cD)) for cH, cV, cD in coeffs[1:]]
    # for i in range(1, levels):
    #     cH, cV, cD = coeffs[i]
    #     norm_coeffs.append((normalize(cH), normalize(cV), normalize(cD)))
    return norm_coeffs



def shift_image_by_offset(img, offset_x, offset_y):
    shifted_img = np.roll(img, shift=offset_y, axis=0)
    shifted_img = np.roll(shifted_img, shift=offset_x, axis=1)
    return shifted_img

def compute_dwt_image(img, levels, wavelet='haar'):
    img = img - img.mean()
    max_level = pywt.dwt_max_level(min(img.shape))
    levels = min(levels, max_level)
    coeffs = pywt.wavedec2(img, wavelet='haar', level=levels)
    dwt_img, _ = pywt.coeffs_to_array(coeffs)
    norm_dwt_img, _ = pywt.coeffs_to_array(normalize_coeffs(coeffs))

    return normalize(dwt_img), normalize(norm_dwt_img)


def main():
    st.title("Shift Image via W/A/S/D + Wavelet Decomposition")

    # Initialize session-state offsets if needed
    if "offset_x" not in st.session_state:
        st.session_state["offset_x"] = 0
    if "offset_y" not in st.session_state:
        st.session_state["offset_y"] = 0

    # -- SIDEBAR CONTROLS --
    with st.sidebar:
        # Sample images
        sample_images = {
            "Camera": data.camera(),
            "Astronaut (gray)": data.astronaut()[..., 0],
            "Checkerboard": data.checkerboard(),
            "Coins": data.coins()
        }

        selected_sample = st.selectbox("Select a Sample Image", list(sample_images.keys()))
        levels = st.slider("DWT Levels", min_value=1, max_value=11, value=3)

        st.markdown("### Shift the image")

        # Row 1: Up in the center
        row_up = st.columns([1, 1, 1], gap="small")
        with row_up[1]:
            if st.button("U"):
                st.session_state["offset_y"] -= 1

        # Row 2: Left - Reset - Right
        row_middle = st.columns([1, 1, 1], gap="small")
        with row_middle[0]:
            if st.button("L"):
                st.session_state["offset_x"] -= 1
        with row_middle[1]:
            if st.button("x"):
                st.session_state["offset_x"] = 0
                st.session_state["offset_y"] = 0
        with row_middle[2]:
            if st.button("R"):
                st.session_state["offset_x"] += 1

        # Row 3: Down in the center
        row_down = st.columns([1, 1, 1], gap="small")
        with row_down[1]:
            if st.button("D"):
                st.session_state["offset_y"] += 1
        # Display current offsets
        st.markdown(
            f"X = {st.session_state['offset_x']}, "
            f"Y = {st.session_state['offset_y']}"
        )

    # Retrieve the selected image
    img = sample_images[selected_sample]
    

    # Apply the shift
    shifted_img = shift_image_by_offset(img, st.session_state["offset_x"], st.session_state["offset_y"])
    shifted_img_float = shifted_img / 255.0
    wavelet_img, norm_wavelet_img = compute_dwt_image(shifted_img_float, levels)

    # Show images in tabs
    tab1, tab2, tab3 = st.tabs(["Shifted Image", "Wavelet Coeffs", "Normalized Wavelet Coeffs"])

    with tab1:
        st.image(shifted_img, caption="Shifted Image", use_container_width=True)

    with tab2:
        st.image(wavelet_img, caption="Wavelet Decomposition", use_container_width=True)

    with tab3:
        st.image(norm_wavelet_img, caption="Wavelet Decomposition", use_container_width=True)



if __name__ == "__main__":
    main()
