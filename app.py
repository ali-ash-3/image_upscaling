from PIL import Image
import torch
import numpy as np
from realesrgan import RealESRGANer

# Try to import RRDBNet from possible locations
try:
    from realesrgan.archs.rrdbnet_arch import RRDBNet
except ImportError:
    from basicsr.archs.rrdbnet_arch import RRDBNet

# === Load image ===
image_path = 'istockphoto-1445597021-612x612.jpg'  # Replace with your input file
image = Image.open(image_path).convert('RGB')

# === Set device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Define model architecture ===
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

# === Load Real-ESRGANer ===
upscaler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4.pth',  # Will auto-download if not available
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# === Predict (Upscale + Denoise + Sharpen) ===
sr_image, _ = upscaler.enhance(np.array(image), outscale=4)
sr_image = Image.fromarray(sr_image)

# === Save the output ===
output_path = 'upscaled.jpg'
sr_image.save(output_path)

print(f"[+] Upscaled image saved to '{output_path}'")
