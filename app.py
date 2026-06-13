import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time
import io
import cv2
import tempfile
import subprocess
from core.model import DCENet, apply_curves
import gc
import os

st.set_page_config(page_title="Zero-DCE | Low-Light Vision", page_icon="🌙", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1 { color: #00FFCC; font-weight: 800; }
    .stSpinner > div > div { border-color: #00FFCC transparent transparent transparent; }
    </style>
""", unsafe_allow_html=True)

st.title("🌙 Zero-Reference Low-Light Vision")
st.markdown("### Powered by DCE-Net (Unsupervised Deep Learning)")

NUM_ITERS = 8

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCENet().to(device)
    model.load_state_dict(torch.load("zero_dce_best-updated.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

@torch.no_grad()
def enhance_image_tensor(tensor_img, model, device):
    """Core enhancement math to be used by both images and video frames"""
    tensor_img = tensor_img.to(device)
    curve_maps = model(tensor_img)
    enhanced_tensor = apply_curves(tensor_img, curve_maps).clamp(0, 1)
    return enhanced_tensor

def process_single_image(image: Image.Image):
    max_size = 1200
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    start_time = time.time()
    tensor_img = T.ToTensor()(image).unsqueeze(0)
    enhanced_tensor = enhance_image_tensor(tensor_img, model, device)
    latency = (time.time() - start_time) * 1000

    enhanced_np = (enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(enhanced_np), latency

def process_video(video_file):
    """Handles frame extraction, enhancement, and FFmpeg conversion with Cloud Memory limits."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_cloud_width = 640
    if orig_width > max_cloud_width:
        scale = max_cloud_width / orig_width
        process_width = max_cloud_width
        process_height = int(orig_height * scale)
    else:
        process_width = orig_width
        process_height = orig_height

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = out_file.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (process_width, process_height))

    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        if orig_width > max_cloud_width:
            frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor_img = T.ToTensor()(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            enhanced_tensor = enhance_image_tensor(tensor_img, model, device)
            
        enhanced_np = (enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        progress = int(((i + 1) / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing Frame {i+1}/{total_frames}...")

        
        del tensor_img, enhanced_tensor, enhanced_np, frame_rgb, frame_bgr
        if i % 10 == 0:
            gc.collect()

    cap.release()
    out.release()
    
    status_text.text("Converting codec for web playback...")
    converted_path = out_path.replace('.mp4', '_h264.mp4')
    
    import subprocess
    command = f"ffmpeg -y -i {out_path} -vcodec libx264 {converted_path}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    total_time = time.time() - start_time
    avg_fps = total_frames / total_time
    
    return converted_path, total_time, avg_fps

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3596/3596091.png", width=60)
st.sidebar.markdown("**Hardware Setup:** " + ("🟢 GPU Acceleration Active" if torch.cuda.is_available() else "🟡 CPU Mode Active (Slow)"))

tab1, tab2 = st.tabs(["🖼️ Image Enhancement", "🎬 Video Enhancement"])

with tab1:
    img_file = st.file_uploader("Upload a dark image...", type=["jpg", "jpeg", "png"], key="img_uploader")
    if img_file is not None:
        original_image = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Low-Light")
            st.image(original_image, width="stretch")
            
        with col2:
            st.markdown("#### Enhanced Output")
            with st.spinner("Executing 8x Light Enhancement..."):
                enhanced_image, latency = process_single_image(original_image)
                st.image(enhanced_image, width="stretch")
                
        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Image Resolution", f"{enhanced_image.width} x {enhanced_image.height}")
        m_col2.metric("Network Parameters", "~0.08 M")
        m_col3.metric("Inference Latency", f"{latency:.1f} ms")

        buf = io.BytesIO()
        enhanced_image.save(buf, format="PNG")
        st.download_button("💾 Download Enhanced Image", buf.getvalue(), "Zero_DCE_Enhanced.png", "image/png", width="stretch")


with tab2:
    st.info("💡 **Tip:** For the best experience on CPU, upload short clips (5-10 seconds) at 720p or lower.")
    vid_file = st.file_uploader("Upload a dark video clip...", type=["mp4", "mov", "avi"], key="vid_uploader")
    
    if vid_file is not None:
        if st.button("🚀 Start Video Processing", width="stretch"):
            with st.spinner("Initializing Video Pipeline..."):
                final_video_path, total_time, avg_fps = process_video(vid_file)
                
            st.success("✅ Video Processing Complete!")
            st.divider()
     
            st.markdown("#### 🎬 Comparison")
            vid_col1, vid_col2 = st.columns(2)
            
            with vid_col1:
                st.markdown("**Original Low-Light**")
                st.video(vid_file)
                
            with vid_col2:
                st.markdown("**Enhanced Output (Zero-DCE)**")
                st.video(final_video_path)

            st.divider()
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Total Processing Time", f"{total_time:.1f} seconds")
            m_col2.metric("Average Processing Speed", f"{avg_fps:.1f} Frames Per Second")
            
            with open(final_video_path, 'rb') as f:
                video_bytes = f.read()
                
            st.download_button(
                label="💾 Download Enhanced Video",
                data=video_bytes,
                file_name="Zero_DCE_Enhanced.mp4",
                mime="video/mp4",
                width="stretch"
            )
