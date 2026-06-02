import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2 
from PIL import Image
from model import UIR_PolyKernel

# Configuration
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 
MODEL_PATH = 'model_checkpoint.pth'

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Robust weight loading
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
        
    model.eval()
    return model, device

model, device = get_model()

# UI Layout
st.title("OrcaCV: Underwater Image Enhancement")
uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- Robust Pre-processing ---
    # Ensure image is converted to RGB (strips Alpha channel if present)
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    # Normalize and prepare input tensor
    input_data = np.array(display_img).astype(np.float32) / 255.0
    input_tensor = torch.tensor(input_data).permute(2,0,1).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # --- Post-processing Pipeline ---
    # Clamp to [0, 1] range and convert back to uint8
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    img_uint8 = (output_cpu * 255).astype(np.uint8)

    # Soft CLAHE (Local Contrast)
    img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.merge((l_enhanced, a, b))
    final_img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Original Input")
    with col2:
        st.image(final_img_rgb, caption="OrcaCV Natural Restoration")
