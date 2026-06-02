import streamlit as st
import torch
import numpy as np
import gdown
import os
import cv2  # This will now work because of step 1
from PIL import Image, ImageOps, ImageFilter
from model import UIR_PolyKernel

# Ensure FILE_ID is a string with quotes!
FILE_ID = '1ZYaHF9LSDH-GFt5W_aTeVPgLXhol_7pT'
MODEL_PATH = 'model_checkpoint.pth'

@st.cache_resource
def get_model():
    # Download model from Drive if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        # fuzzy=True helps handle Google Drive's redirect pages
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UIR_PolyKernel().to(device)
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, device

# Initialize model
model, device = get_model()

# 2. User Interface
st.title("OrcaCV: Underwater Image Enhancement")
st.write("Upload a murky underwater image to see real-time enhancement.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Pre-processing
    img = Image.open(uploaded_file).convert('RGB')
    display_img = img.copy()
    display_img.thumbnail((512, 512)) # Maintains aspect ratio
    
    # Prepare input tensor
    input_tensor = torch.tensor(np.array(display_img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-processing Pipeline
    # 1. Clamp and convert to 8-bit
    output_cpu = torch.clamp(output.cpu(), 0, 1).squeeze(0).permute(1,2,0).numpy()
    img_uint8 = (output_cpu * 255).astype('uint8')

    # 2. Apply CLAHE (Spatial Domain Contrast Enhancement)
    img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.merge((l_enhanced, a, b))
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)

    # 3. Apply Unsharp Masking (Frequency Domain Edge Sharpening)
    final_img = Image.fromarray(img_rgb).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Original Input")
    with col2:
        st.image(final_img, caption="OrcaCV Enhanced")
