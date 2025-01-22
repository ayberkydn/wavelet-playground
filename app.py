import streamlit as st
from skimage import data
from utils import shift_img, compute_dwt

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
        
        # Let the user select the wavelet type
        wavelets = [
            'db1', 'db2', 'db3', 
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

        # Display current offsets
        st.markdown(
            f"**Offset X** = {st.session_state['offset_x']:.2f}, "
            f"**Offset Y** = {st.session_state['offset_y']:.2f}"
        )

    # Retrieve the selected image
    img = sample_images[selected_sample]
    
    # Apply the shift (assumes shift_img can handle float offsets properly)
    shifted_img = shift_img(
        img, 
        shift_x=st.session_state["offset_x"], 
        shift_y=st.session_state["offset_y"]
    )
    
    # Convert to float for wavelet transform
    shifted_img_float = shifted_img / 255.0
    
    # Compute wavelet decomposition with selected wavelet
    wavelet_img, norm_wavelet_img = compute_dwt(shifted_img_float, levels, wavelet=selected_wavelet)

    # Show images in tabs
    tab1, tab2, tab3 = st.tabs(["Shifted Image", "Wavelet Coeffs", "Normalized Wavelet Coeffs"])

    with tab1:
        st.image(shifted_img, caption="Shifted Image", use_container_width=True)

    with tab2:
        st.image(wavelet_img, caption="Wavelet Decomposition", use_container_width=True)

    with tab3:
        st.image(norm_wavelet_img, caption="Wavelet Decomposition (Normalized)", use_container_width=True)

if __name__ == "__main__":
    main()