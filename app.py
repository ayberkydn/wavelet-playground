import streamlit as st
from skimage import data
from utils import compute_dwt, scale, shift_rotate_img, coeff_to_img, plot_histogram, reconstruct_dwt, vis_coeffs
from skimage.transform import resize
from streamlit_vertical_slider import vertical_slider
import pywt
import numpy as np


def main():
    st.set_page_config(
        page_title="Wavelet Playground ",
        layout="wide"
    )
    st.title("Wavelet Playground")

    # ----------------- Session State for shifting/rotating -------------
    if "dx" not in st.session_state:
        st.session_state["dx"] = 0.0
    if "dy" not in st.session_state:
        st.session_state["dy"] = 0.0
    if "rot" not in st.session_state:
        st.session_state["rot"] = 0.0

    # ------------------------- Sidebar ------------------------------
    with st.sidebar:
        sample_images = {
            "Astronaut (gray)":  data.astronaut()[..., 0],
            "Cat":               data.cat()[..., 0],
            "Checkerboard":      data.checkerboard(),
            "Coins":             data.coins(),  
            "Grass":             data.grass(),
            "Gravel":            data.gravel(),
            "Logo":              data.logo()[..., 0],
            "Text":              data.text(),
            "Rocket (gray)":     data.rocket()[..., 0],
        }
        selected_sample = st.selectbox("Select a Sample Image", list(sample_images.keys()))
        
        selected_family = st.selectbox("Select a Wavelet", pywt.families()[:7], index=0)
        selected_wavelet = st.selectbox("Select a Wavelet", pywt.wavelist(family=selected_family), index=0)
     
        # levels = st.slider("DWT Levels", min_value=1, max_value=8, value=8)

        # SHIFT CONTROLS
        st.markdown("### Shift")
        shift_step = st.radio("Shift step (pixels)", [0.5, 1, 5], index=0, horizontal=True)
        row_up = st.columns([1, 1, 1], gap="small")
        row_mid = st.columns([1, 1, 1], gap="small")
        row_down = st.columns([1, 1, 1], gap="small")

        with row_up[0]:
            if st.button("↖", key="up-left"):
                st.session_state["dy"] -= shift_step
                st.session_state["dx"] -= shift_step
        with row_up[1]:
            if st.button("↑", key="up"):
                st.session_state["dy"] -= shift_step
        with row_up[2]:
            if st.button("↗", key="up-right"):
                st.session_state["dy"] -= shift_step
                st.session_state["dx"] += shift_step
        with row_mid[0]:
            if st.button("←", key="left"):
                st.session_state["dx"] -= shift_step
        with row_mid[1]:
            if st.button("0", key="reset"):
                st.session_state["dx"] = 0.0
                st.session_state["dy"] = 0.0
        with row_mid[2]:
            if st.button("→", key="right"):
                st.session_state["dx"] += shift_step
        with row_down[0]:
            if st.button("↙", key="down-left"):
                st.session_state["dy"] += shift_step
                st.session_state["dx"] -= shift_step
        with row_down[1]:
            if st.button("↓", key="down"):
                st.session_state["dy"] += shift_step
        with row_down[2]:
            if st.button("↘", key="down-right"):
                st.session_state["dy"] += shift_step
                st.session_state["dx"] += shift_step

        # ROTATION CONTROLS
        st.markdown("### Rotate")
        rotation_row = st.columns([1, 1, 1, 1, 1], gap="small")
        with rotation_row[0]:
            if st.button("-5°", key="rotate_left2"):
                st.session_state["rot"] += 5
        with rotation_row[1]:
            if st.button("-1°", key="rotate_left3"):
                st.session_state["rot"] += 1
        with rotation_row[2]:
            if st.button("0", key="reset_rotation"):
                st.session_state["rot"] = 0
        with rotation_row[3]:
            if st.button("1°", key="rotate_right1"):
                st.session_state["rot"] -= 1
        with rotation_row[4]:
            if st.button("5°", key="rotate_right2"):
                st.session_state["rot"] -= 5
        
        st.markdown(
            f"**X** = {st.session_state['dx']:.2f} | "
            f"**Y** = {st.session_state['dy']:.2f}"
        )
        st.markdown(f"**Angle** = {st.session_state['rot']:.1f}°")

    # --------------------- Main Image and DWT -----------------------
    # Resize sample image to a consistent shape, e.g. 512x512
    base_img = resize(sample_images[selected_sample], (512, 512))

    # Shift / rotate
    base_img = shift_rotate_img(
        base_img,
        shift_x=st.session_state["dx"],
        shift_y=st.session_state["dy"],
        angle=st.session_state["rot"]
    )
    base_img = scale(base_img, min_val=-1, max_val=1)

    # Wavelet decomposition
    wavelet_img, norm_wavelet_img, coeffs, cHs, cVs, cDs = compute_dwt(
        base_img,
        wavelet=selected_wavelet
    )


    # --------------- Display in Tabs --------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Image", 
        "Wavelet Decomposition", 
        "Wavelet Coefficients",
        "Reconstruction",
    ])

    with tab1:
        st.image(scale(base_img, min_val=0, max_val=1), caption="Image", use_container_width=True)

    with tab2:
        # Create two sub-tabs
        tab21, tab22 = st.tabs(["Raw Decomposition", "Scaled Decomposition"])
        with tab21:
            # Show the wavelet decomposition image unscaled
            st.image(
                scale(wavelet_img, min_val=0, max_val=1), 
                caption="Raw Wavelet Decomposition", 
                use_container_width=True
                )

        with tab22:
            # Show the wavelet decomposition image scaled to [0, 1]
            st.image(
                scale(norm_wavelet_img, min_val=0, max_val=1), 
                caption="Scaled Wavelet Decomposition (approx = 0)", 
                use_container_width=True
            )

    with tab3:
        slider_coeff = st.slider("Level", min_value=1, max_value=len(coeffs)-1, value=1, key="coeff_slider")
        r1, r2, r3 = st.columns([1, 1, 1], gap="small")
        with r1:
            show_img = coeff_to_img(cHs[slider_coeff-1], (512, 512))
            show_img = shift_rotate_img(show_img, shift_x=st.session_state["dx"], shift_y=st.session_state["dy"], angle=st.session_state["rot"], reverse=True)
            show_img = scale(show_img, min_val=0, max_val=1)
            # st.write(f"Coeff shape: {cHs[slider_coeff-1].shape}")
            # st.write(f"Coeff min: {cHs[slider_coeff-1].min():.2f}")
            # st.write(f"Coeff max: {cHs[slider_coeff-1].max():.2f}")
            # st.write(f"Coeff mean: {cHs[slider_coeff-1].mean():.2f}")
            # st.write(f"Coeff std: {cHs[slider_coeff-1].std():.2f}")
            # st.write(f"Coeff median: {np.median(cHs[slider_coeff-1]):.2f}")

            st.image(show_img, caption="Horizontal Coefficients", use_container_width=True)
            st.plotly_chart(plot_histogram(cHs[slider_coeff-1], title=""))
        with r2:
            show_img = coeff_to_img(cVs[slider_coeff-1], (512, 512))
            show_img = shift_rotate_img(show_img, shift_x=st.session_state["dx"], shift_y=st.session_state["dy"], angle=st.session_state["rot"], reverse=True)
            show_img = scale(show_img, min_val=0, max_val=1)
            st.write(f"Coeff shape: {cVs[slider_coeff-1].shape}")
            st.image(show_img, caption="Vertical Coefficients", use_container_width=True)
            st.plotly_chart(plot_histogram(cVs[slider_coeff-1], title=""))
        with r3:
            show_img = coeff_to_img(cDs[slider_coeff-1], (512, 512))
            show_img = shift_rotate_img(show_img, shift_x=st.session_state["dx"], shift_y=st.session_state["dy"], angle=st.session_state["rot"], reverse=True)
            show_img = scale(show_img, min_val=0, max_val=1)
            st.write(f"Coeff shape: {cDs[slider_coeff-1].shape}")
            st.image(show_img, caption="Diagonal Coefficients", use_container_width=True)
            st.plotly_chart(plot_histogram(cDs[slider_coeff-1], title=""))

    with tab4:
        vis_levels = st.pills("Levels", options=list(range(0, len(coeffs))), default=[], selection_mode="multi")
        show_img = vis_coeffs(coeffs, selected_levels=vis_levels, wavelet=selected_wavelet)
        show_img = shift_rotate_img(show_img, shift_x=st.session_state["dx"], shift_y=st.session_state["dy"], angle=st.session_state["rot"], reverse=True)
        show_img = scale(show_img, min_val=0, max_val=1)
        st.image(show_img, caption="Reconstructed Image", use_container_width=True)

if __name__ == "__main__":
    main()