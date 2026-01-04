from PIL import Image
import os

# Source image path (generated artifact)
source_path = "/home/ftferreira02/.gemini/antigravity/brain/57b09b29-30d5-4f7a-b778-ba7d8421a092/phishing_shield_logo_1766839986489.png"
output_dir = "extension/icons"

# Ensure output directory exists (already checked but good practice)
os.makedirs(output_dir, exist_ok=True)

try:
    with Image.open(source_path) as img:
        # Save 128x128
        img.resize((128, 128), Image.Resampling.LANCZOS).save(os.path.join(output_dir, "icon128.png"))
        # Save 48x48
        img.resize((48, 48), Image.Resampling.LANCZOS).save(os.path.join(output_dir, "icon48.png"))
        # Save 16x16
        img.resize((16, 16), Image.Resampling.LANCZOS).save(os.path.join(output_dir, "icon16.png"))
        
    print("Icons generated successfully.")
except Exception as e:
    print(f"Error generating icons: {e}")
