import streamlit as st
from skimage import data
from utils import compute_dwt, scale, transform_img, coeff_to_img, plot_histogram, reconstruct_from_coeffs, quantize_dwt, plot_image, threshold_dwt
from skimage.transform import resize
import pywt
import numpy as np


def main():
    st.set_page_config(
        page_title="Wavelet Playground ",
        layout="wide"
    )
    st.title("Wavelet Playground")

    shape = (512, 512)

    # ----------------- Session State for shifting/rotating/scaling -------------
    if "dx" not in st.session_state:
        st.session_state["dx"] = 0.0
    if "dy" not in st.session_state:
        st.session_state["dy"] = 0.0
    if "rot" not in st.session_state:
        st.session_state["rot"] = 0.0
    if "scale_factor" not in st.session_state:
        st.session_state["scale_factor"] = 1.0

    # ------------------------- Sidebar ------------------------------
    with st.sidebar:
        sample_images = {
            "Astronaut":  data.astronaut(),
            "Cat":        data.cat(),
        }
        selected_sample = st.selectbox("Image", list(sample_images.keys()))
        
        selected_family = st.selectbox("Wavelet Family", pywt.families()[1:7], index=3)
        wavelet_index = {
            "bior": 12,
            "coif": 5,
            "db": 0,
            "sym": 0,
            "rbio": 0,
            "dmey": 0,
        }
        selected_wavelet = st.selectbox("Wavelet", pywt.wavelist(family=selected_family), index=wavelet_index[selected_family])
        use_plotly = st.toggle("Use Plotly Images", value=True)
        show_histograms = st.toggle("Show Histograms", value=True)


        
        base_img = resize(sample_images[selected_sample], shape)
        base_img = transform_img(
            base_img,
            shift_x=st.session_state["dx"],
            shift_y=st.session_state["dy"],
            angle=st.session_state["rot"],
            scale_factor=st.session_state["scale_factor"]
        )
        base_img = scale(base_img, min_val=-1, max_val=1)        

        wavelet_img, norm_wavelet_img, coeffs = compute_dwt(
            base_img,
            wavelet=selected_wavelet
        )

        # SHIFT, ROTATE AND SCALE CONTROLS
        shift_controls, rotate_controls, scale_controls = st.tabs(["Shift", "Rotate", "Scale"])
        
        with shift_controls:
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

            st.markdown(
                f"**X** = {st.session_state['dx']:.2f} | "
                f"**Y** = {st.session_state['dy']:.2f}"
            )

        with rotate_controls:
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
            
            st.markdown(f"**Angle** = {st.session_state['rot']:.1f}°")

        with scale_controls:
            scale_row = st.columns([1, 1, 1, 1, 1], gap="small")
            with scale_row[0]:
                if st.button("×0.5", key="scale_half"):
                    st.session_state["scale_factor"] *= 0.5
            with scale_row[1]:
                if st.button("×0.9", key="scale_down"):
                    st.session_state["scale_factor"] *= 0.9
            with scale_row[2]:
                if st.button("×1", key="reset_scale"):
                    st.session_state["scale_factor"] = 1.0
            with scale_row[3]:
                if st.button("×1.1", key="scale_up"):
                    st.session_state["scale_factor"] *= 1.1
            with scale_row[4]:
                if st.button("×2", key="scale_double"):
                    st.session_state["scale_factor"] *= 2.0
            
            st.markdown(f"**Scale** = ×{st.session_state['scale_factor']:.2f}")

        quantization_settings, thresholding_settings, display_settings = st.tabs(["Quantization", "Threshold", "Display"])
 
        with quantization_settings:
            st.subheader("Quantization Parameters")
            bits_per_level = {}
            for lvl in range(1, len(coeffs[0])):
                bits_per_level[lvl] = st.slider(
                    f"Bits (Level {lvl})",
                    min_value=2.0,
                    max_value=8.0,
                    value=8.0,
                    step=0.5,
                )
        with thresholding_settings:
            st.subheader("Thresholding Parameters")
            threshold = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=0.2,  
                value=0.01,
                step=0.01,
            )
        
    coeffs = quantize_dwt(coeffs, bits_per_level=bits_per_level)    
    coeffs = threshold_dwt(coeffs, threshold=threshold)
    cHs = [coeff[0] for coeff in coeffs[0][1:]]
    cVs = [coeff[1] for coeff in coeffs[0][1:]]
    cDs = [coeff[2] for coeff in coeffs[0][1:]]

    rec_tab, decomp_tab, coeffs_tab = st.tabs([
        "Reconstruction", 
        "Wavelet Decomposition", 
        "Wavelet Coefficients",
    ])

    with rec_tab:
        vis_levels = st.pills("Levels", options=list(range(0, len(coeffs[0]))), default=list(range(0, len(coeffs[0]))), selection_mode="multi")
        rec = reconstruct_from_coeffs(coeffs, selected_levels=vis_levels, wavelet=selected_wavelet)
        rec_error = np.abs(base_img - rec)
        rec = scale(rec, min_val=0, max_val=1)

        tab1, tab2, tab3 = st.tabs(["Image", "Reconstruction", "Reconstruction Error"])
        with tab1:
            st.markdown("",help=f"""***Max***: {base_img.max():.2f}  
            ***Min***: {base_img.min():.2f}  
            ***Mean***: {base_img.mean():.2f}  
            ***Std Dev***: {base_img.std():.2f}
            """)
            img_to_show = scale(base_img, min_val=0, max_val=1)
            if use_plotly:
                st.plotly_chart(plot_image(img_to_show, title="Original Image"), use_container_width=True)
            else:
                st.image(img_to_show, caption="Original Image", use_container_width=True)
        with tab2:
            st.markdown("",help=f"""***Max***: {rec.max():.2f}  
            ***Min***: {rec.min():.2f}  
            ***Mean***: {rec.mean():.2f}  
            ***Std Dev***: {rec.std():.2f}
            """)
            if use_plotly:
                st.plotly_chart(plot_image(rec, title="Reconstructed Image"), use_container_width=True)
            else:
                st.image(rec, caption="Reconstructed Image", use_container_width=True)
        with tab3:
            st.markdown("",help=f"""***Max***: {rec_error.max():.2f}  
            ***Min***: {rec_error.min():.2f}  
            ***Mean***: {rec_error.mean():.2f}  
            ***Std Dev***: {rec_error.std():.2f}
            """)
            error_to_show = scale(rec_error, min_val=0, max_val=1)
            if use_plotly:
                st.plotly_chart(plot_image(error_to_show, title="Reconstruction Error"), use_container_width=True)
            else:
                st.image(error_to_show, caption="Reconstruction Error", use_container_width=True)

    with decomp_tab:
        tab1, tab2 = st.tabs(["Raw Decomposition", "Scaled Decomposition"])
        with tab1:
            img_to_show = scale(wavelet_img, min_val=0, max_val=1)
            if use_plotly:
                st.plotly_chart(plot_image(img_to_show, title="Raw Wavelet Decomposition"), use_container_width=True)
            else:
                st.image(img_to_show, caption="Raw Wavelet Decomposition", use_container_width=True)

        with tab2:
            img_to_show = scale(norm_wavelet_img, min_val=0, max_val=1)
            if use_plotly:
                st.plotly_chart(plot_image(img_to_show, title="Scaled Wavelet Decomposition"), use_container_width=True)
            else:
                st.image(img_to_show, caption="Scaled Wavelet Decomposition", use_container_width=True)

    with coeffs_tab:
        slider_coeff = st.slider("Level", min_value=1, max_value=len(coeffs[0])-1, value=1, key="coeff_slider")
        r1, r2, r3 = st.columns([1, 1, 1], gap="small")
        with r1:
            show_img = coeff_to_img(cHs[slider_coeff-1], shape)
            show_img = scale(show_img, min_val=0, max_val=1)
            st.markdown(f"", help=f"***Shape***: {cHs[slider_coeff-1].shape} \n\n ***Min***: {cHs[slider_coeff-1].min():.2f} \n\n ***Max***: {cHs[slider_coeff-1].max():.2f} \n\n ***Mean***: {cHs[slider_coeff-1].mean():.2f} \n\n ***Std***: {cHs[slider_coeff-1].std():.2f} \n\n ***Median***: {np.median(cHs[slider_coeff-1]):.2f}")
            if use_plotly:
                st.plotly_chart(plot_image(show_img, title="cH"), use_container_width=True)
            else:
                st.image(show_img, caption="cH", use_container_width=True)
            if show_histograms:
                st.plotly_chart(plot_histogram(cHs[slider_coeff-1], title=""))
        with r2:
            show_img = coeff_to_img(cVs[slider_coeff-1], shape)
            show_img = scale(show_img, min_val=0, max_val=1)
            st.markdown(f"", help=f"***Shape***: {cVs[slider_coeff-1].shape} \n\n ***Min***: {cVs[slider_coeff-1].min():.2f} \n\n ***Max***: {cVs[slider_coeff-1].max():.2f} \n\n ***Mean***: {cVs[slider_coeff-1].mean():.2f} \n\n ***Std***: {cVs[slider_coeff-1].std():.2f} \n\n ***Median***: {np.median(cVs[slider_coeff-1]):.2f}")
            if use_plotly:
                st.plotly_chart(plot_image(show_img, title="cV"), use_container_width=True)
            else:
                st.image(show_img, caption="cV", use_container_width=True)
            if show_histograms:
                st.plotly_chart(plot_histogram(cVs[slider_coeff-1], title=""))
        with r3:
            show_img = coeff_to_img(cDs[slider_coeff-1], shape)
            show_img = scale(show_img, min_val=0, max_val=1)
            st.markdown(f"", help=f"***Shape***: {cDs[slider_coeff-1].shape} \n\n ***Min***: {cDs[slider_coeff-1].min():.2f} \n\n ***Max***: {cDs[slider_coeff-1].max():.2f} \n\n ***Mean***: {cDs[slider_coeff-1].mean():.2f} \n\n ***Std***: {cDs[slider_coeff-1].std():.2f} \n\n ***Median***: {np.median(cDs[slider_coeff-1]):.2f}")
            if use_plotly:
                st.plotly_chart(plot_image(show_img, title="cD"), use_container_width=True)
            else:
                st.image(show_img, caption="cD", use_container_width=True)
            if show_histograms:
                st.plotly_chart(plot_histogram(cDs[slider_coeff-1], title=""))

if __name__ == "__main__":
    main()