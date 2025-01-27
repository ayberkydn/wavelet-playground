import streamlit as st
from skimage import data
from utils import compute_dwt, normalize, shift_rotate_img, create_wavelet_coefficients_video
from skimage.transform import resize

def main():
    st.set_page_config(
        page_title="Wavelet Playground - Resized Frames for Video",
        layout="wide"
    )
    st.title("Wavelet Playground with Resized Video Frames")

    # ----------------- Session State for shifting/rotating -------------
    if "offset_x" not in st.session_state:
        st.session_state["offset_x"] = 0.0
    if "offset_y" not in st.session_state:
        st.session_state["offset_y"] = 0.0
    if "rotation_angle" not in st.session_state:
        st.session_state["rotation_angle"] = 0.0

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
        
        wavelets = ['haar', 'db2', 'db3']
        selected_wavelet = st.selectbox("Select a Wavelet", wavelets, index=0)

        levels = st.slider("DWT Levels", min_value=1, max_value=8, value=3)

        # SHIFT CONTROLS
        st.markdown("### Shift the image")
        shift_step = st.slider("Shift step (pixels)", 0.5, 2.0, 0.5)
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

        # ROTATION CONTROLS
        st.markdown("### Rotate the image")
        rotation_row = st.columns([1, 1, 1, 1, 1, 1, 1], gap="small")
        with rotation_row[0]:
            if st.button("<-", key="rotate_left1"):
                st.session_state["rotation_angle"] += 30
        with rotation_row[1]:
            if st.button("<-", key="rotate_left2"):
                st.session_state["rotation_angle"] += 5
        with rotation_row[2]:
            if st.button("<-", key="rotate_left3"):
                st.session_state["rotation_angle"] += 1
        with rotation_row[3]:
            if st.button("0", key="reset_rotation"):
                st.session_state["rotation_angle"] = 0
        with rotation_row[4]:
            if st.button("->", key="rotate_right1"):
                st.session_state["rotation_angle"] -= 1
        with rotation_row[5]:
            if st.button("->", key="rotate_right2"):
                st.session_state["rotation_angle"] -= 5
        with rotation_row[6]:
            if st.button("->", key="rotate_right3"):
                st.session_state["rotation_angle"] -= 50

        st.markdown(
            f"**X** = {st.session_state['offset_x']:.2f} | "
            f"**Y** = {st.session_state['offset_y']:.2f}"
        )
        st.markdown(f"**Angle** = {st.session_state['rotation_angle']:.1f}°")

    # --------------------- Main Image and DWT -----------------------
    # Resize sample image to a consistent shape, e.g. 512x512
    base_img = resize(sample_images[selected_sample], (512, 512))

    # Shift / rotate
    base_img = shift_rotate_img(
        base_img,
        shift_x=st.session_state["offset_x"],
        shift_y=st.session_state["offset_y"],
        angle=st.session_state["rotation_angle"]
    )
    base_img = normalize(base_img)  # scale to [0,1]

    # Wavelet decomposition
    wavelet_img, coeffs = compute_dwt(
        base_img,
        levels=levels,
        wavelet=selected_wavelet
    )

    # Replace the call to create_horizontal_detail_video with the new function
    wavelet_video = create_wavelet_coefficients_video(coeffs, fps=3, target_size=(512, 512))

    # --------------- Display in Tabs --------------
    tab1, tab2, tab3 = st.tabs([
        "Transformed Image", 
        "Wavelet Decomposition", 
        "Video: Wavelet Coefficients"
    ])

    with tab1:
        st.image(base_img, caption="Shifted + Rotated Image", use_container_width=True)

    with tab2:
        st.image(wavelet_img, caption="Wavelet Decomposition (approx = 0)", use_container_width=True)

    with tab3:
        if wavelet_video:
            st.video(wavelet_video, format="video/mp4")
        else:
            st.warning("No detail frames. Try increasing the number of DWT levels.")

if __name__ == "__main__":
    main()
