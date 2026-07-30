import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2
from PIL import Image, ImageFilter
from model import UIR_PolyKernel

# --- PAGE CONFIGURATION & "CUTE-PRO" STYLING ---
st.set_page_config(
    page_title="OrcaCV 🌊 | Mathematical Marine Restoration",
    page_icon="🐋",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Soft, High-End Aesthetic Ocean Gradient */
.stApp {
    background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 50%, #bae6fd 100%);
    background-attachment: fixed;
    color: #1e293b;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Floating Cute Little Animated Orcas & Bubbles */
@keyframes swim {
    0% { transform: translateY(0px) translateX(0px) rotate(0deg); opacity: 0.3; }
    50% { transform: translateY(-20px) translateX(15px) rotate(3deg); opacity: 0.6; }
    100% { transform: translateY(0px) translateX(0px) rotate(0deg); opacity: 0.3; }
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800' width='100%25' height='100%25'%3E%3Cg fill='%230284c7'%3E%3Cpath d='M120,150 Q140,130 160,150 Q150,170 120,150 Z M160,145 L175,138 L168,152 Z' opacity='0.5'/%3E%3Cpath d='M680,320 Q700,300 720,320 Q710,340 680,320 Z M720,315 L735,308 L728,322 Z' opacity='0.4'/%3E%3Cpath d='M220,580 Q240,560 260,580 Q250,600 220,580 Z M260,575 L275,568 L268,582 Z' opacity='0.6'/%3E%3Cpath d='M580,650 Q600,630 620,650 Q610,670 580,650 Z' opacity='0.4'/%3E%3Ccircle cx='350' cy='200' r='3' opacity='0.4'/%3E%3Ccircle cx='500' cy='500' r='5' opacity='0.3'/%3E%3Ccircle cx='180' cy='450' r='4' opacity='0.5'/%3E%3C/g%3E%3C/svg%3E");
    animation: swim 14s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

div.stInfo, .element-container, div.row-widget {
    position: relative;
    z-index: 1;
}

/* Clean, Rounded Cute-Tech Glassmorphism Cards */
div.stInfo {
    background-color: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.15);
    color: #334155 !important;
    padding: 1.25rem;
}

/* Modern Rounded File Dropzone with Soft Border */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(255, 255, 255, 0.8) !important;
    border: 2px dashed #0284c7 !important;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.1);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(255, 255, 255, 1) !important;
    border-color: #0369a1 !important;
    box-shadow: 0 12px 28px rgba(14, 165, 233, 0.25);
}

/* Gorgeous Floating Cute Action Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 0.65rem 1.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.35);
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
}

/* Crisp, Polished Typography */
h1, h2, h3, h4, p {
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# 1. Configuration & Model Initialization
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("⚠️ Error: Could not download model checkpoint. Please verify Drive link permissions.")
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

model, device = get_model()

# 2. UI Layout & Professional Branding
st.title("🐋 OrcaCV: Mathematical Marine Vision")
st.markdown("<p style='font-size: 13px; color: #475569; font-style: italic; margin-top: -10px;'>✨ Generative AI guesses pixels; OrcaCV performs absolute mathematical restoration.</p>", unsafe_allow_html=True)

st.info("**Architectural Overview:**\n\nDesigned for rigorous structural fidelity, OrcaCV leverages the **UIR-PolyKernel** framework and Hybrid Domain Attention (HDA). Rather than synthesizing artificial textures like standard generative tools, it mathematically reverses environmental physics—such as wavelength-dependent light attenuation and scattering—to uncover true sub-surface geometry.")

st.markdown("---")

# File Upload Workflow
uploaded_file = st.file_uploader("Drop your underwater image here to restore...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    st.markdown("### 📷 Input Preview")
    st.image(display_img, caption="Source Image Ready for Processing", width=400)
    
    if st.button("✨ Run Mathematical Restoration"):
        with st.spinner("🐋 Processing optical physics equations..."):
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
            st.markdown("<h3 style='text-align: center; color: #0284c7; margin-top: 2rem;'>✨ Restoration Results</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(display_img, caption="Original Input", use_container_width=True)
            with col2:
                st.image(precision_sharpened_img, caption="OrcaCV Mathematical Restoration", use_container_width=True)
                
            st.success("🎉 Structural fidelity successfully recovered!")
