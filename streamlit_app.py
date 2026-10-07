import streamlit as st
from PIL import Image
import pandas as pd
import io

from utils.client import LlamaServerClient, DEFAULT_SERVER_URL, DEFAULT_SYSTEM_PROMPT

try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except ImportError:
    HAS_CROPPER = False

st.set_page_config(
    page_title="NVIDIA LocateEverything UI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 NVIDIA LocateEverything / LocateAnything UI")
st.caption("Interactive Vision-Grounding & Object Bounding Box Visualizer for llama_server")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Server Configuration")
    server_url = st.text_input("llama_server URL", value=DEFAULT_SERVER_URL)
    
    if st.button("🔌 Check Connection", use_container_width=True):
        is_ok, msg = LlamaServerClient.check_health(server_url)
        if is_ok:
            st.success(f"🟢 {msg}")
        else:
            st.error(f"🔴 {msg}")

    st.divider()
    st.header("🎨 Annotation Style")
    box_thickness = st.slider("Box Line Thickness", min_value=1, max_value=10, value=3)
    font_size = st.slider("Label Font Size", min_value=10, max_value=32, value=14)
    fill_alpha = st.slider("Fill Transparency (Alpha)", min_value=0, max_value=255, value=45)
    temperature = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=80)

# Main UI Layout
col_left, col_right = st.columns([1.1, 1.2])

with col_left:
    st.subheader("📥 1. Upload & Crop Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    final_input_image = None

    if uploaded_file is not None:
        try:
            raw_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            raw_image = None

        if raw_image is not None:
            enable_crop = st.checkbox("✂️ Crop input image before asking LocateEverything", value=False)
            
            if enable_crop:
                crop_methods = []
                if HAS_CROPPER:
                    crop_methods.append("Interactive Canvas (Mouse Drag)")
                crop_methods.append("Coordinate Sliders (% Boundary)")
                
                crop_mode = st.radio("Cropping Method", crop_methods, horizontal=True)

                if HAS_CROPPER and crop_mode == "Interactive Canvas (Mouse Drag)":
                    aspect_choice = st.selectbox(
                        "Aspect Ratio Preset", 
                        ["Freeform", "1:1 Square", "4:3 Standard", "16:9 Widescreen"]
                    )
                    aspect_dict = {
                        "Freeform": None,
                        "1:1 Square": (1, 1),
                        "4:3 Standard": (4, 3),
                        "16:9 Widescreen": (16, 9)
                    }
                    st.write(" Drag and resize the box on the image below:")
                    cropped_image = st_cropper(
                        raw_image,
                        realtime_update=True,
                        box_color="#38bdf8",
                        aspect_ratio=aspect_dict[aspect_choice],
                        key="interactive_cropper"
                    )
                else:
                    st.write("Adjust crop boundaries (% of width / height):")
                    c1, c2 = st.columns(2)
                    with c1:
                        x_range = st.slider("X Range (Left to Right %)", 0, 100, (0, 100))
                    with c2:
                        y_range = st.slider("Y Range (Top to Bottom %)", 0, 100, (0, 100))

                    img_w, img_h = raw_image.size
                    left = int(img_w * x_range[0] / 100.0)
                    right = max(left + 1, int(img_w * x_range[1] / 100.0))
                    top = int(img_h * y_range[0] / 100.0)
                    bottom = max(top + 1, int(img_h * y_range[1] / 100.0))

                    cropped_image = raw_image.crop((left, top, right, bottom))

                w_c, h_c = cropped_image.size
                st.info(f"✂️ **Cropped Region:** {w_c} x {h_c} px (Original: {raw_image.width} x {raw_image.height} px)")
                st.image(cropped_image, caption="Selected Cropped Part for LocateEverything", use_column_width=True)
                final_input_image = cropped_image
            else:
                st.image(raw_image, caption=f"Full Input Image ({raw_image.width} x {raw_image.height} px)", use_column_width=True)
                final_input_image = raw_image

    st.subheader("❓ 2. Vision Grounding Prompt")
    
    # Store quick prompt in session state if button clicked
    if "prompt_val" not in st.session_state:
        st.session_state["prompt_val"] = "locate building"

    col_qp1, col_qp2, col_qp3, col_qp4 = st.columns(4)
    if col_qp1.button("🏢 Buildings", use_container_width=True):
        st.session_state["prompt_val"] = "locate building"
    if col_qp2.button("🚗 Vehicles", use_container_width=True):
        st.session_state["prompt_val"] = "locate vehicle"
    if col_qp3.button("👤 People", use_container_width=True):
        st.session_state["prompt_val"] = "locate person"
    if col_qp4.button("🔍 All Objects", use_container_width=True):
        st.session_state["prompt_val"] = "locate all objects in image"

    prompt_input = st.text_input(
        "Question / Prompt",
        key="prompt_val",
        placeholder="e.g. locate building, locate car, find all objects"
    )

    run_btn = st.button("🚀 Run Vision Grounding", type="primary", use_container_width=True)

with col_right:
    st.subheader("🖼️ 3. LocateEverything Output")

    if run_btn:
        if final_input_image is None:
            st.warning("⚠️ Please upload an image first.")
        else:
            with st.spinner("Analyzing specified image region with LocateEverything..."):
                annotated_img, detections, raw_response, exec_time, error_msg = LlamaServerClient.predict(
                    server_url=server_url,
                    image=final_input_image,
                    prompt=prompt_input,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    box_thickness=box_thickness,
                    font_size=font_size,
                    fill_alpha=fill_alpha
                )

            if error_msg:
                st.error(f"❌ Error: {error_msg}")
                if raw_response:
                    with st.expander("Raw Server Output"):
                        st.code(raw_response)
            else:
                st.success(f"⚡ Inference completed in **{exec_time}s** | Found **{len(detections)}** object(s)")
                
                tabs = st.tabs(["🖼️ Annotated Result", "📷 Analyzed Image Input"])
                with tabs[0]:
                    st.image(annotated_img, caption="Annotated Image with Coordinates", use_column_width=True)
                with tabs[1]:
                    st.image(final_input_image, caption="Exact Region Sent to Model", use_column_width=True)

                st.subheader("📊 Detected Bounding Boxes")
                if detections:
                    df_rows = []
                    for idx, d in enumerate(detections, start=1):
                        norm_box_str = f"[{d['norm_box'][0]}, {d['norm_box'][1]}, {d['norm_box'][2]}, {d['norm_box'][3]}]"
                        pixel_box_str = f"[{d['pixel_box'][0]}, {d['pixel_box'][1]}, {d['pixel_box'][2]}, {d['pixel_box'][3]}]" if 'pixel_box' in d else "-"
                        df_rows.append({
                            "#": idx,
                            "Label": d["label"],
                            "Normalized Box [ymin, xmin, ymax, xmax]": norm_box_str,
                            "Pixel Box [xmin, ymin, xmax, ymax]": pixel_box_str,
                            "Area %": f"{d.get('area_pct', 0.0)}%"
                        })
                    st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
                else:
                    st.info("No bounding boxes were detected in the response text.")

                with st.expander("📜 Raw Model Response Text"):
                    st.code(raw_response)
