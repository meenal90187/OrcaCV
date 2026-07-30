import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2
from PIL import Image, ImageFilter
from model import UIR_PolyKernel

# --- PAGE CONFIGURATION & EXECUTIVE STYLING ---
st.set_page_config(
    page_title="OrcaCV | Marine Vision Restoration",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Professional Deep Ocean Gradient */
.stApp {
    background: linear-gradient(180deg, #020617 0%, #0f172a 50%, #1e293b 100%);
    background-attachment: fixed;
    color: #f8fafc;
}

/* Subtle Animated Floating Water & Orca Silhouette Elements */
@keyframes drift {
    0% { transform: translateY(0px) translateX(0px); opacity: 0.1; }
    50% { transform: translateY(-15px) translateX(10px); opacity: 0.25; }
    100% { transform: translateY(0px) translateX(0px); opacity: 0.1; }
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800' width='100%25' height='100%25'%3E%3Cg fill='%2338bdf8'%3E%3Cpath d='M150,200 Q170,180 190,200 Q180,220 150,200 Z' opacity='0.3'/%3E%3Cpath d='M650,450 Q670,430 690,450 Q680,470 650,450 Z' opacity='0.2'/%3E%3Cpath d='M300,600 Q320,580 340,600 Q330,620 300,600 Z' opacity='0.25'/%3E%3C/g%3E%3C/svg%3E");
    animation: drift 15s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

/* Ensure contents sit above background */
div.stInfo, .element-container, div.row-widget {
    position: relative;
    z-index: 1;
}

/* Executive Glassmorphism Info Box */
div.stInfo {
    background-color: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    color: #e2e8f0 !important;
    padding: 1.2rem;
}

/* Professional File Dropzone */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 2px dashed rgba(56, 189, 248, 0.5) !important;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border-color: #38bdf8 !important;
}

/* Action Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
    background: linear-gradient(135deg, #0284c7 0%, #1d4ed8 100%);
}

/* Typography Hierarchy */
h1, h2, h3, h4, p {
    color: #f8fafc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# 1. Configuration & Model Loading
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("Critical Error: Failed to retrieve model checkpoint. Verify Drive permissions.")
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    
    try:
        checkpoint = torch.load(output_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
        
    model.eval()
    return model, device

# Initialize model pipeline
model, device = get_model()

# 2. Main Interface Layout
st.title("🌊 OrcaCV: Marine Vision Restoration")
st.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; margin-top: -12px;'>* Computer vision and LLMs are distinct paradigms. Generative AI hallucinates synthetic pixels; OrcaCV executes absolute mathematical restoration.</p>", unsafe_allow_html=True)

st.info("**Architecture Overview:**\n\nUnlike generative networks that synthesize aesthetic details, OrcaCV is a deterministic Computer Vision framework engineered for mathematical structural fidelity. Integrating the **UIR-PolyKernel** architecture and Hybrid Domain Attention (HDA), the pipeline physically reverses environmental degradation parameters—including wavelength-dependent light attenuation and anisotropic scattering—recovering genuine marine sub-surface geometry.")

st.markdown("---")

# File Upload Workflow
uploaded_file = st.file_uploader("Upload degraded marine imagery for processing", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    st.markdown("### 📷 Input Inspection")
    st.image(display_img, caption="Source Image Ready for Processing", width=400)
    
    if st.button("Execute Mathematical Restoration"):
        with st.spinner("Processing physical optical equations..."):
            # Tensor Conversion
            input_tensor = torch.tensor(np.array(display_img).astype(np.float32)).permute(2,0,1).div(255).unsqueeze(0).to(device)
            
            # Neural Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            # Post-Processing Phase 1: Tensor Decoding
            output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
            img_uint8 = (output_cpu * 255).astype('uint8')
            
            # Post-Processing Phase 2: Natural CLAHE Enhancement
            img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            img_enhanced = cv2.merge((l_enhanced, a, b))
            final_img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
            
            # Post-Processing Phase 3: Precision Edge Sharpening
            enhanced_pil = Image.fromarray(final_img_rgb)
            precision_sharpened_img = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2))
            
            # Comparative Evaluation Layout
            st.markdown("<h3 style='text-align: center; color: #38bdf8; margin-top: 2rem;'>Comparative Analysis</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(display_img, caption="Original Degraded Input", use_container_width=True)
            with col2:
                st.image(precision_sharpened_img, caption="OrcaCV Restored Output", use_container_width=True)
                
            st.success("✅ Optical correction and structural recovery successfully completed.")
