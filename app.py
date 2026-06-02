import streamlit as st
import torch
import numpy as np
import gdown
import os
from PIL import Image, ImageOps
from model import UIR_PolyKernel
from scipy.ndimage import median_filter

# 1. Download and Load model (Combined for safety)
@st.cache_resource
def get_model():
    # Replace with your actual File ID
    FILE_ID = 1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT
    output_path = 'model_checkpoint.pth'
    
    # Download if not already there
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, output_path, quiet=False)
    
    # Load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

model, device = get_model()

# 2. UI
st.title("OrcaCV: Underwater Image Enhancement")
uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    resized_img = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
    
    input_tensor = torch.tensor(np.array(resized_img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    filtered_img = median_filter(output_cpu, size=3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_img, caption="Input")
    with col2:
        st.image(filtered_img, caption="OrcaCV Enhanced")
