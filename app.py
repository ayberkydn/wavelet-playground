import streamlit as st
from skimage import data
from skimage.transform import resize
from utils import shift_img, compute_dwt, normalize, rotate_img

# Set page config first (must be done before creating other Streamlit elements)
st.set_page_config(
    page_title="Wavelet Playground",
    layout="wide"
)

def main():
    st.title("Wavelet Playground")

    # Initialize session-state offsets if needed
    if "offset_x" not in st.session_state:
        st.session_state["offset_x"] = 0.0
    if "offset_y" not in st.session_state:
        st.session_state["offset_y"] = 0.0

    # Initialize session-state for rotation
    if "rotation_angle" not in st.session_state:
        st.session_state["rotation_angle"] = 0.0

    with st.sidebar:
        sample_images = {
            "Astronaut (gray)":  data.astronaut()[..., 0],
            # "Binary Blobs":      data.binary_blobs(),
            "Brick":             data.brick(),
            "Colorwheel":        data.colorwheel()[..., 0],
            "Camera":            data.camera(),
            "Cat":               data.cat()[..., 0],
            "Checkerboard":      data.checkerboard(),
            "Clock":             data.clock(),
            "Coffee":            data.coffee()[..., 0],
            "Coins":             data.coins(),
            # "Eagle":             data.eagle(),
            "Grass":             data.grass(),
            "Gravel":            data.gravel(),
            "Horse":             data.horse(),
            "Logo":              data.logo()[..., 0],
            "Page":              data.page(),
            "Text":              data.text(),
            "Rocket (gray)":     data.rocket()[..., 0],
        }

        selected_sample = st.selectbox("Select a Sample Image", list(sample_images.keys()))
        
        # Let the user select the wavelet type
        wavelets = [
            'haar', 'db2', 'db3', 
        ]
        selected_wavelet = st.selectbox("Select a Wavelet", wavelets, index=0)

        levels = st.slider("DWT Levels", min_value=1, max_value=11, value=3)

        st.markdown("### Shift the image")

        # Select shift step (fractional value)
        shift_step = st.slider("Shift step (pixels)", 0.1, 5.0, 0.5, 0.1)

        row_up = st.columns([1, 1, 1], gap="small")
        row_mid = st.columns([1, 1, 1], gap="small")
        row_down = st.columns([1, 1, 1], gap="small")

        with row_up[0]:
            if st.button("↖", key="up-left"):
                st.session_state["offset_y"] -= shift_step
                st.session_state["offset_x"] -= shift_step
        with row_up[1]:
            if st.button("↑", key="up"):
                st.session_state["offset_y"] -= shift_step
        with row_up[2]:
            if st.button("↗", key="up-right"):
                st.session_state["offset_y"] -= shift_step
                st.session_state["offset_x"] += shift_step

        with row_mid[0]:
            if st.button("←", key="left"):
                st.session_state["offset_x"] -= shift_step
        with row_mid[1]:
            if st.button("0", key="reset"):
                st.session_state["offset_x"] = 0.0
                st.session_state["offset_y"] = 0.0
        with row_mid[2]:
            if st.button("→", key="right"):
                st.session_state["offset_x"] += shift_step
        
        with row_down[0]:
            if st.button("↙", key="down-left"):
                st.session_state["offset_y"] += shift_step
                st.session_state["offset_x"] -= shift_step
        with row_down[1]:
            if st.button("↓", key="down"):
                st.session_state["offset_y"] += shift_step
        with row_down[2]:
            if st.button("↘", key="down-right"):
                st.session_state["offset_y"] += shift_step
                st.session_state["offset_x"] += shift_step

        st.markdown(
            f"**X** = {st.session_state['offset_x']:.2f} \n\n"
            f"**Y** = {st.session_state['offset_y']:.2f}"
        )


        # --- Rotation Controls ---
        rotation_row = st.columns([1, 1], gap="small")
        st.markdown("### Rotate the image")
        with rotation_row[0]:
            if st.button("<-", key="rotate_right"):
                st.session_state["rotation_angle"] += 1
        with rotation_row[1]:
            if st.button("->", key="rotate_left"):
                st.session_state["rotation_angle"] -= 1
        if st.button("Reset rotation"):
            st.session_state["rotation_angle"] = 0.0

        st.markdown(f"**Angle** = {st.session_state['rotation_angle']:.1f}°")

    # Resize chosen image
    img = resize(sample_images[selected_sample], (1024, 1024))

    # Apply shift
    img = shift_img(
        img,
        shift_x=st.session_state["offset_x"],
        shift_y=st.session_state["offset_y"]
    )
    
    # Apply rotation
    img = rotate_img(img, st.session_state["rotation_angle"])

    # Normalize
    img = normalize(img)

    # Compute wavelet decomposition with selected wavelet
    wavelet_img, norm_wavelet_img = compute_dwt(img, levels, wavelet=selected_wavelet)

    # Show images in tabs
    tab1, tab2, tab3 = st.tabs(["Transformed Image", "Wavelet Coeffs", "Normalized Wavelet Coeffs"])

    with tab1:
        st.image(img, caption="Shifted + Rotated Image", use_container_width=True)

    with tab2:
        st.image(wavelet_img, caption="Wavelet Decomposition", use_container_width=True)

    with tab3:
        st.image(norm_wavelet_img, caption="Wavelet Decomposition (Normalized)", use_container_width=True)

if __name__ == "__main__":
    main()
