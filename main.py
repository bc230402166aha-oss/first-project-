import json
import cv2
import numpy as np
import streamlit as st
import torch


def process_raster_with_overlay(image_bytes, filename):
    """Processes input image bytes, generates segmentation mask overlay,

    and calculates spatial metrics.
    """
    # 1. Convert uploaded file bytes into OpenCV image format
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to read image file.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Check GPU Availability
    device_used = "cuda" if torch.cuda.is_available() else "cpu"

    # --- INFERENCE STEP ---
    # Replace the section below with your actual model inference
    # Example:
    # tensor_input = transform(image_rgb).to(device_used)
    # mask = model(tensor_input).squeeze().cpu().numpy()

    # Placeholder mask for demonstration:
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    # Simulating detected region
    mask[int(h * 0.3) : int(h * 0.6), int(w * 0.3) : int(w * 0.6)] = 1
    # ----------------------

    # 3. Calculate Spatial Metrics
    flood_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    flood_coverage_pct = round((flood_pixels / total_pixels) * 100, 2)
    confidence_score = 0.94  # Replace with actual model output score

    # 4. Generate Visual Overlay (Fixed Blending Logic)
    overlay = image_rgb.copy()
    blue_color = np.array([0, 120, 255], dtype=np.uint8)

    # Safely blend blue tint onto detected regions without OpenCV shape mismatch
    mask_boolean = mask == 1
    overlay[mask_boolean] = (
        0.5 * overlay[mask_boolean] + 0.5 * blue_color
    ).astype(np.uint8)

    # 5. Construct JSON Response
    metrics = {
        "filename": filename,
        "status": "success",
        "device_used": device_used,
        "spatial_metrics": {
            "estimated_flood_coverage_pct": flood_coverage_pct,
            "confidence_score": confidence_score,
        },
    }

    return metrics, overlay


# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="Spatial Intelligence Engine",
    page_icon="🗺️",
    layout="wide",
)

st.title("Spatial Intelligence Engine")
st.caption("Kaggle Cloud GPU Accelerated Feature Extraction")

# File Upload Input
uploaded_file = st.file_uploader(
    "Choose File", type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")

    if st.button("Process Raster", type="primary"):
        with st.spinner("Processing raster on GPU..."):
            try:
                # Read bytes directly in memory
                image_bytes = uploaded_file.read()
                metrics, overlay_img = process_raster_with_overlay(
                    image_bytes, uploaded_file.name
                )

                # Layout Side-by-Side Results
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Visual Mask Overlay")
                    st.image(
                        overlay_img,
                        caption="Detected Flood Overlay (Highlighted Blue)",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("Extraction Metrics")
                    st.json(metrics)

            except Exception as e:
                st.error(f"Error processing raster: {str(e)}")