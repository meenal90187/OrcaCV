import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2
from PIL import Image, ImageOps, ImageFilter
from model import UIR_PolyKernel

# 1. Configuration
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT' 

@st.cache_resource
def get_model():
    output_path = 'model_checkpoint.pth'
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False, fuzzy=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    checkpoint = torch.load(output_path, map_location=device)
    
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
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) 
    
    # Inference
    input_tensor = torch.tensor(np.array(display_img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-processing
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    enhanced_img = Image.fromarray((output_cpu * 255).astype('uint8'))
    
    # Apply CLAHE for better local contrast
    img_cv = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_cv = cv2.merge((l,a,b))
    enhanced_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB))
    
    # Apply Unsharp Mask for edge clarity
    final_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Original Input")
    with col2:
        st.image(final_img, caption="OrcaCV Enhanced (Clearer)")
