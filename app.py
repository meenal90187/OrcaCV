import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2
from PIL import Image, ImageFilter
from model import UIR_PolyKernel

# --- UI PRETTIFICATION & CUSTOM CSS ---
st.set_page_config(page_title="OrcaCV Restoration", page_icon="🌊", layout="centered")

st.markdown("""
<style>
/* Deep Ocean Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0284c7 100%);
    background-attachment: fixed;
    color: #f8fafc;
}

/* Floating Little Orcas Background Effect via SVG Patterns */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800' width='100%25' height='100%25' opacity='0.08'%3E%3Cpath fill='%23ffffff' d='M100,200 Q120,180 140,200 Q130,220 100,200 Z'/%3E%3Cpath fill='%23ffffff' d='M600,500 Q620,480 640,500 Q630,520 600,500 Z'/%3E%3Cpath fill='%23ffffff' d='M300,700 Q320,680 340,700 Q330,720 300,700 Z'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

/* Glassmorphism containers for text & components */
div.stInfo, .css-1dp5vir {
    background-color: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    color: #f1f5f9 !important;
}

/* Clean, highly visible drag-and-drop file uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(15, 23, 42, 0.6);
    border: 2px dashed #38bdf8;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(4px);
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(30, 41, 59, 0.8);
    border-color: #7dd3fc;
}

/* Typography adjustments for dark theme contrast */
h1, h2, h3, h4, p {
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

# 1. Configuration
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("Error: Could not download the model. Check Google Drive link permissions.")
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

# Initialize model
model, device = get_model()

# 2. UI Layout & Typography
st.title("🌊 OrcaCV: Marine Vision Restoration")
st.markdown("<p style='font-size: 13px; color: #cbd5e1; font-style: italic; margin-top: -15px;'>* Computer vision and LLMs are not twins. Generative AI hallucinates fake pixels; we perform absolute mathematical restoration.</p>", unsafe_allow_html=True)

st.info("**About the Architecture:**\n\nUnlike modern Generative models that 'hallucinate' data to make an image look pretty, OrcaCV is a pure Computer Vision framework designed for strict structural fidelity. By utilizing the **UIR-PolyKernel** architecture and Hybrid Domain Attention (HDA), this model mathematically reverses real-world optical physics—such as wavelength-dependent attenuation and anisotropic light scattering.")

st.markdown("---")

# Styled File Uploader
uploaded_file = st.file_uploader("Drop your degraded underwater image here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Instant Preview before running heavy calculations
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    st.markdown("### 📷 Image Preview")
    st.image(display_img, caption="Uploaded Input Ready for Processing", width=400)
    
    if st.button("✨ Run Mathematical Restoration", type="primary"):
        with st.spinner("Mathematically restoring optical physics..."):
            # Convert to Tensor
            input_tensor = torch.tensor(np.array(display_img).astype(np.float32)).permute(2,0,1).div(255).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            # Post-processing Phase 1: Raw Output
            output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
            img_uint8 = (output_cpu * 255).astype('uint8')
            
            # Post-processing Phase 2: Natural CLAHE
            img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            img_enhanced = cv2.merge((l_enhanced, a, b))
            final_img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
            
            # Convert back to PIL & Sharpen
            enhanced_pil = Image.fromarray(final_img_rgb)
            precision_sharpened_img = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2))
            
            # Display Results Side by Side
            st.markdown("<h3 style='text-align: center; color: #38bdf8;'>Restoration Results</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(display_img, caption="Original Input", use_container_width=True)
            with col2:
                st.image(precision_sharpened_img, caption="OrcaCV Mathematical Restoration", use_container_width=True)
                
            st.success("✅ Structural fidelity successfully recovered!")
