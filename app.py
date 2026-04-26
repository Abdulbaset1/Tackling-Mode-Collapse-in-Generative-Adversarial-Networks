import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import requests
from pathlib import Path
import tempfile
import os
import sys

# Set page config
st.set_page_config(
    page_title="GAN Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Display Python version for debugging
st.sidebar.markdown(f"Python version: {sys.version.split()[0]}")

# Define the model architecture (same as in training)
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

# Model configuration
class CFG:
    Z_DIM = 100
    IMAGE_SIZE = 64

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Alternative download URLs - you need to update these with the correct raw URLs
# To get the correct URL: Go to your release, right-click on the .pt file, and copy the link address
MODEL_URLS = {
    "DCGAN": "https://github.com/Abdulbaset1/Tackling-Mode-Collapse-in-Generative-Adversarial-Networks/releases/download/v1_dcgan/dcgan_ckpt_epoch_60.pt",
    "WGAN-GP": "https://github.com/Abdulbaset1/Tackling-Mode-Collapse-in-Generative-Adversarial-Networks/releases/download/v1_wgan/wgangp_ckpt_epoch_20.pt"
}

@st.cache_resource
def load_model(model_type):
    """Load pre-trained model from GitHub Releases"""
    try:
        checkpoint_url = MODEL_URLS[model_type]
        st.info(f"Downloading {model_type} model checkpoint...")
        
        # Download checkpoint file with headers to avoid rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(checkpoint_url, stream=True, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Failed to download checkpoint. Status code: {response.status_code}")
            st.error(f"URL attempted: {checkpoint_url}")
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_bar.progress(min(1.0, downloaded / total_size))
            
            progress_bar.progress(1.0)
            checkpoint_path = tmp_file.name
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            st.warning(f"Failed to load with weights_only=False, trying without...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model
        model = Generator(CFG.Z_DIM).to(device)
        
        # Load weights (handle different checkpoint structures)
        if 'G' in checkpoint:
            model.load_state_dict(checkpoint['G'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Clean up temp file
        os.unlink(checkpoint_path)
        
        st.success(f"✅ {model_type} model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

def generate_images(model, num_images=4, z_dim=100):
    """Generate images using the model"""
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_images, z_dim, 1, 1, device=device)
        
        # Generate images
        fake_images = model(z).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        fake_images = torch.clamp(fake_images, 0, 1)
        
        # Convert to PIL images
        pil_images = []
        for i in range(num_images):
            img = fake_images[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        return pil_images

def main():
    st.title("🎨 GAN Image Generator")
    st.markdown("Generate Pokémon-style images using DCGAN and WGAN-GP models")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Settings")
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["DCGAN", "WGAN-GP"],
        help="DCGAN: Standard GAN with BatchNorm | WGAN-GP: Wasserstein GAN with Gradient Penalty"
    )
    
    num_images = st.sidebar.slider(
        "Number of Images to Generate",
        min_value=1,
        max_value=16,
        value=4,
        step=1
    )
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.model_type = None
    
    # Load model button
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner(f"Loading {model_choice} model..."):
            model = load_model(model_choice)
            if model is not None:
                st.session_state.model = model
                st.session_state.model_type = model_choice
                st.session_state.model_loaded = True
                st.rerun()
    
    # Check if model is loaded
    if st.session_state.model_loaded and st.session_state.model is not None:
        st.success(f"✅ {st.session_state.model_type} model is ready!")
        
        # Add a random seed slider for reproducibility
        seed = st.sidebar.number_input("Random Seed (optional)", min_value=0, max_value=9999, value=42, step=1)
        
        # Generation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"🎨 Generate {num_images} Images", type="primary", use_container_width=True):
                # Set random seed for reproducibility
                if seed:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                
                with st.spinner("Generating images..."):
                    generated_images = generate_images(
                        st.session_state.model, 
                        num_images=num_images,
                        z_dim=CFG.Z_DIM
                    )
                    st.session_state.generated_images = generated_images
        
        # Display generated images
        if 'generated_images' in st.session_state and st.session_state.generated_images:
            st.markdown("---")
            st.header("✨ Generated Images")
            
            # Display images in a responsive grid
            cols_per_row = min(4, num_images)
            rows = (num_images + cols_per_row - 1) // cols_per_row
            
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    img_idx = row * cols_per_row + col_idx
                    if img_idx < len(st.session_state.generated_images):
                        with cols[col_idx]:
                            st.image(st.session_state.generated_images[img_idx], 
                                   caption=f"Image {img_idx+1}", 
                                   use_container_width=True)
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Download Images")
            
            # Option to download all as a zip
            if st.button("📦 Download All as ZIP", type="secondary"):
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, img in enumerate(st.session_state.generated_images):
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="PNG")
                        zip_file.writestr(f"generated_{st.session_state.model_type}_image_{idx+1}.png", img_buffer.getvalue())
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer,
                    file_name=f"gan_generated_images_{st.session_state.model_type}.zip",
                    mime="application/zip"
                )
            
            # Individual downloads in a grid
            st.markdown("**Or download individually:**")
            download_cols = st.columns(min(4, num_images))
            for idx, img in enumerate(st.session_state.generated_images):
                col_idx = idx % len(download_cols)
                with download_cols[col_idx]:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label=f"⬇️ Image {idx+1}",
                        data=byte_im,
                        file_name=f"gan_{st.session_state.model_type}_{idx+1}.png",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
    else:
        st.info("👈 **Get Started:** Load a model from the sidebar to start generating images!")
        
        # Display model information
        with st.expander("📖 About the Models", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎯 DCGAN (Deep Convolutional GAN)
                - Standard GAN architecture using convolutional layers
                - Trained for 60 epochs
                - Uses Batch Normalization and LeakyReLU activations
                - **Pros:** Fast training, good image quality
                - **Cons:** Can suffer from mode collapse
                """)
            
            with col2:
                st.markdown("""
                ### 🌊 WGAN-GP (Wasserstein GAN with Gradient Penalty)
                - Improved GAN training with Wasserstein distance
                - Trained for 20 epochs
                - Uses Gradient Penalty for Lipschitz constraint
                - **Pros:** More stable training, better mode coverage
                - **Cons:** Slower training, more computationally intensive
                """)
            
            st.markdown("""
            ### 📊 Training Dataset
            Both models were trained on **Pokémon sprite images** (64x64 resolution).
            
            ### 🔧 How to use:
            1. Select a model (DCGAN or WGAN-GP)
            2. Click "Load Model" (downloads pre-trained weights from GitHub)
            3. Adjust number of images and random seed
            4. Click "Generate Images" to create new Pokémon-style images
            5. Download individual images or all as a ZIP file
            """)

if __name__ == "__main__":
    main()
