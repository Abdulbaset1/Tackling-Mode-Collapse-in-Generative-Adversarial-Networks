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

# Set page config
st.set_page_config(
    page_title="GAN Image Generator",
    page_icon="🎨",
    layout="wide"
)

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

@st.cache_resource
def load_model(model_type, checkpoint_url):
    """Load pre-trained model from GitHub Releases"""
    try:
        # Download checkpoint file
        st.info(f"Downloading {model_type} model checkpoint...")
        response = requests.get(checkpoint_url, stream=True)
        
        if response.status_code != 200:
            st.error(f"Failed to download checkpoint from {checkpoint_url}")
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            checkpoint_path = tmp_file.name
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
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
    
    # GitHub Release URLs (raw file URLs)
    # Note: You need to use raw.githubusercontent.com URLs or GitHub's release download URLs
    model_urls = {
        "DCGAN": "https://github.com/Abdulbaset1/Tackling-Mode-Collapse-in-Generative-Adversarial-Networks/releases/download/v1_dcgan/dcgan_ckpt_epoch_60.pt",
        "WGAN-GP": "https://github.com/Abdulbaset1/Tackling-Mode-Collapse-in-Generative-Adversarial-Networks/releases/download/v1_wgan/wgangp_ckpt_epoch_20.pt"
    }
    
    # Load model based on selection
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner(f"Loading {model_choice} model..."):
            model = load_model(model_choice, model_urls[model_choice])
            st.session_state['model'] = model
            st.session_state['model_type'] = model_choice
    
    # Check if model is loaded
    if 'model' in st.session_state and st.session_state['model'] is not None:
        st.success(f"✅ {st.session_state['model_type']} model is ready!")
        
        # Generation button
        if st.button(f"Generate {num_images} Images", type="primary", use_container_width=True):
            with st.spinner("Generating images..."):
                generated_images = generate_images(
                    st.session_state['model'], 
                    num_images=num_images,
                    z_dim=CFG.Z_DIM
                )
                st.session_state['generated_images'] = generated_images
        
        # Display generated images
        if 'generated_images' in st.session_state:
            st.markdown("---")
            st.header("Generated Images")
            
            # Display images in a grid
            cols = st.columns(min(num_images, 4))
            for idx, img in enumerate(st.session_state['generated_images']):
                col_idx = idx % len(cols)
                with cols[col_idx]:
                    st.image(img, caption=f"Image {idx+1}", use_container_width=True)
            
            # Download button for individual images
            st.markdown("---")
            st.subheader("Download Images")
            
            for idx, img in enumerate(st.session_state['generated_images']):
                # Convert PIL to bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label=f"Download Image {idx+1}",
                    data=byte_im,
                    file_name=f"generated_{st.session_state['model_type']}_image_{idx+1}.png",
                    mime="image/png",
                    key=f"download_{idx}"
                )
    else:
        st.info("👈 Please load a model from the sidebar to start generating images!")
        
        # Display model information
        with st.expander("📖 About the Models"):
            st.markdown("""
            ### DCGAN (Deep Convolutional GAN)
            - Standard GAN architecture using convolutional layers
            - Trained for 60 epochs
            - Uses Batch Normalization and LeakyReLU activations
            
            ### WGAN-GP (Wasserstein GAN with Gradient Penalty)
            - Improved GAN training with Wasserstein distance
            - Trained for 20 epochs
            - More stable training and better mode coverage
            - Uses Gradient Penalty for Lipschitz constraint
            
            ### Training Dataset
            Both models were trained on Pokémon sprite images.
            """)

if __name__ == "__main__":
    main()
