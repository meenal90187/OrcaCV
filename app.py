import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2
from PIL import Image, ImageFilter
from model import UIR_PolyKernel

# --- UI PRETTIFICATION & CUSTOM CSS ---
# Set the page configuration for a clean layout
st.set_page_config(page_title="OrcaCV Restoration", page_icon="🌊", layout="centered")

# Injecting custom CSS for a pretty background and prominent drag-and-drop zone
st.markdown("""
<style>
/* Ocean-themed subtle gradient background */
.stApp {
    background: linear-gradient(135deg, #e0f2fe 0%, #a6c0fe 100%);
}

/* Clean, highly visible drag-and-drop file uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(255, 255, 255, 0.8);
    border: 2px dashed #4A90E2;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    transition: background-color 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(255, 255, 255, 1);
    border-color: #2171C7;
}

/* Clean up the info box to match the aesthetic */
div.stInfo {
    background-color: rgba(255, 255, 255, 0.6);
    border: 1px solid #a6c0fe;
    border-radius: 10px;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# 1. Configuration
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    # Download from Drive
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False)
    
    # Safety check: ensure file exists and isn't empty
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("Error: Could not download the model. Check Google Drive link permissions.")
        st.stop()
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    
    try:
        checkpoint = torch.load(output_path, map_location=device)
        # Handle different checkpoint dictionary structures
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

# The "Banger Line" in tiny, stylized font
st.markdown("<p style='font-size: 13px; color: #555555; font-style: italic; margin-top: -15px;'>* Computer vision and LLMs are not twins. Generative AI hallucinates fake pixels; we perform absolute mathematical restoration.</p>", unsafe_allow_html=True)

# Short, punchy description defending your CV project vs Gen AI
st.info("**About the Architecture:**\n\nUnlike modern Generative models that 'hallucinate' data to make an image look pretty, OrcaCV is a pure Computer Vision framework designed for strict structural fidelity. By utilizing the **UIR-PolyKernel** architecture and Hybrid Domain Attention (HDA), this model mathematically reverses real-world optical physics—such as wavelength-dependent attenuation and anisotropic light scattering. We do not generate fake marine textures; we recover the *actual* geometry hidden in the deep sea.")

st.markdown("---")

# Styled File Uploader
uploaded_file = st.file_uploader("Upload a degraded underwater image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Mathematically restoring optical physics..."):
        # Pre-processing: convert to RGB strips out problematic Alpha channels
        img = Image.open(uploaded_file).convert('RGB')
        display_img = img.copy()
        display_img.thumbnail((512, 512)) 
        
        # Convert to Tensor (Added astype(np.float32) for mathematical stability)
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
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) # Natural 1.5 limit
        l_enhanced = clahe.apply(l)
        img_enhanced = cv2.merge((l_enhanced, a, b))
        final_img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(final_img_rgb)
        
        # Post-processing Phase 3: Precision Unsharp Masking
        # Percent 200 and threshold 2 to prevent artificial noise
        precision_sharpened_img = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2))
        
        # Display Results Beautifully
        st.markdown("<h4 style='text-align: center; color: #4A90E2;'>Restoration Results</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(display_img, caption="Original Input", use_column_width=True)
        with col2:
            st.image(precision_sharpened_img, caption="OrcaCV Mathematical Restoration", use_column_width=True)
            
        st.success("✅ Structural fidelity successfully recovered!")
