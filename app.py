import streamlit as st
import torch
import numpy as np
import gdown
import os
from PIL import Image, ImageOps, ImageFilter
from model import UIR_PolyKernel

# 1. Configuration - Replace with your actual File ID
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    
    # Download from Drive
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False, fuzzy=True)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    checkpoint = torch.load(output_path, map_location=device)
    
    # Handle different checkpoint dictionary structures
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, device

model, device = get_model()

# 2. UI
st.title("OrcaCV: Underwater Image Enhancement")
uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Pre-processing: Maintain Aspect Ratio instead of cropping
    img = Image.open(uploaded_file).convert('RGB')
    
    # Resize while keeping proportions to avoid distortion/blur
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    # Inference
    input_tensor = torch.tensor(np.array(display_img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-processing: Convert to image and Sharpen
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    enhanced_img = Image.fromarray((output_cpu * 255).astype('uint8'))
    
    # Apply sharpening to fix blurriness
    sharpened_img = enhanced_img.filter(ImageFilter.SHARPEN)
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Original Input")
    with col2:
        st.image(sharpened_img, caption="OrcaCV Enhanced")
