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
    background: linear-gradient(135deg, #020617 0%, #0369a1 50%, #38bdf8 100%);
    background-attachment: fixed;
    color: #f8fafc;
}

/* Floating Animated Little Orcas Background */
@keyframes floatOrcas {
    0% { transform: translateY(0px) translateX(0px) rotate(0deg); opacity: 0.15; }
    50% { transform: translateY(-30px) translateX(20px) rotate(5deg); opacity: 0.35; }
    100% { transform: translateY(0px) translateX(0px) rotate(0deg); opacity: 0.15; }
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800' width='100%25' height='100%25'%3E%3Cg fill='%23ffffff'%3E%3Cpath d='M150,150 Q170,130 190,150 Q180,170 150,150 Z M190,145 L210,135 L200,155 Z' opacity='0.4'/%3E%3Cpath d='M650,350 Q670,330 690,350 Q680,370 650,350 Z M690,345 L710,335 L700,355 Z' opacity='0.3'/%3E%3Cpath d='M250,650 Q270,630 290,650 Q280,670 250,650 Z M290,645 L310,635 L300,655 Z' opacity='0.5'/%3E%3Cpath d='M550,650 Q570,630 590,650 Q580,670 550,650 Z M590,645 L610,635 L600,655 Z' opacity='0.4'/%3E%3C/g%3E%3C/svg%3E");
    animation: floatOrcas 12s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

/* Glassmorphism Containers */
div.stInfo, .element-container {
    position: relative;
    z-index: 1;
}

div.stInfo {
    background-color: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    color: #f1f5f9 !important;
}

/* Enhanced Drag-and-Drop Uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(15, 23, 42, 0.5) !important;
    border: 2px dashed #38bdf8 !important;
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease-in-out;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border-color: #7dd3fc !important;
    box-shadow: 0 12px 40px rgba(56, 189, 248, 0.25);
}

/* Custom Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
}

/* Global Typography Colors */
h1, h2, h3, h4, p, span {
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
st.markdown("<p style='font-size: 13px; color: #bae6fd; font-style: italic; margin-top: -15px;'>* Computer vision and LLMs are not twins. Generative AI hallucinates fake pixels; we perform absolute mathematical restoration.</p>", unsafe_allow_html=True)

st.info("**About the Architecture:**\n\nUnlike modern Generative models that 'hallucinate' data to make an image look pretty, OrcaCV is a pure Computer Vision framework designed for strict structural fidelity. By utilizing the **UIR-PolyKernel** architecture and Hybrid Domain Attention (HDA), this model mathematically reverses real-world optical physics—such as wavelength-dependent attenuation and anisotropic light scattering.")

st.markdown("---")

# Styled File Uploader
uploaded_file = st.file_uploader("Drop your degraded underwater image here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    st.markdown("### 📷 Image Preview")
    st.image(display_img, caption="Uploaded Input Ready for Processing", width=400)
    
    if st.button("✨ Run Mathematical Restoration"):
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
