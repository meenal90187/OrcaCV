import streamlit as st
import torch
import numpy as np
import gdown
import os
from PIL import Image, ImageOps, ImageFilter
from model import UIR_PolyKernel

# 1. Configuration
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    # Download from Drive with safety check
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        # Using 'output=' for newer gdown compatibility
        gdown.download(url, output=output_path, quiet=False, fuzzy=True)
    
    # Check if download succeeded
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("Error: Could not download the model. Check Google Drive link permissions.")
        st.stop()
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    
    try:
        checkpoint = torch.load(output_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
        
    model.eval()
    return model, device

model, device = get_model()

# 2. UI
st.title("OrcaCV: Underwater Image Enhancement")
uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    input_tensor = torch.tensor(np.array(display_img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    enhanced_img = Image.fromarray((output_cpu * 255).astype('uint8'))
    sharpened_img = enhanced_img.filter(ImageFilter.SHARPEN)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Original Input")
    with col2:
        st.image(sharpened_img, caption="OrcaCV Enhanced")
