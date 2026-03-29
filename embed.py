import os
import zarr
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F


def embed_single(crop_tensor, model, patch_size=16, embed_mode="pixel"):
    """Passes a single padded image crop through DINOv3."""
    _, _, H, W = crop_tensor.shape
    
    with torch.no_grad():
        outputs = model.forward_features(crop_tensor)
        patch_tokens = outputs["x_norm_patchtokens"]  # Shape: (1, N, Embedding_Dim)
        embedding_dim = patch_tokens.shape[-1]
        
        h_patches, w_patches = H // patch_size, W // patch_size
        
        spatial_embeddings = patch_tokens.reshape(1, h_patches, w_patches, embedding_dim)
        spatial_embeddings = spatial_embeddings.permute(0, 3, 1, 2) 
        
        if embed_mode == "patch":
            return spatial_embeddings
        elif embed_mode == "pixel":
            dense_embeddings = F.interpolate(spatial_embeddings, size=(H, W), mode="bilinear", align_corners=False)
            return dense_embeddings
        else:
            raise ValueError("embed_mode must be 'patch' or 'pixel'")


def embed_all(
    crop_config,
    model,
    data_dir="data",
    embed_mode="pixel",
    patch_size=16,
    device="cpu"
):
    """
    Extracts DINOv3 embeddings for specific JSON-configured crops and saves them to disk.
    """
    
    # Input normalization for DINOv3 backbones
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    extracted_features = {}

    # Process all crops in config
    for volume_name, crops in crop_config.items():
        print(f"\n--- Processing volume: {volume_name} ---")
        
        # Resolve dynamic dataset path
        em_dir = Path(data_dir) / volume_name / f"{volume_name}.zarr" / "recon-1" / "em"
        modality_dirs = list(em_dir.glob("fibsem-*")) + list(em_dir.glob("tem-*"))
        
        if not modality_dirs:
            print(f"  [!] Skipping {volume_name}: No imaging modality folder found.")
            continue
            
        zarr_path = modality_dirs[0] / "s0"  # use highest resolution
        dataset = zarr.open(str(zarr_path), mode='r')
        extracted_features[volume_name] = {}
        
        for crop in crops:
            crop_id = crop["crop_id"]
            z = crop["z"]
            y1, y2 = crop["y_min"], crop["y_max"]
            x1, x2 = crop["x_min"], crop["x_max"]
            
            print(f"  -> Embedding '{crop_id}' (Z:{z}, Y:{y1}-{y2}, X:{x1}-{x2})")
            
            # Slice the numpy array directly from zarr, then normalize
            img_array = torch.from_numpy(dataset[z, y1:y2, x1:x2]).float()
            img_tensor = (img_array - img_array.min()) / (img_array.max() - img_array.min())
                
            # Format to (1, 3, H, W) and normalize
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            img_tensor = (img_tensor - mean) / std
            
            _, _, orig_H, orig_W = img_tensor.shape
            
            # Pad to multiple of patch_size for DINOv3 architecture compatibility
            pad_h = (patch_size - orig_H % patch_size) % patch_size
            pad_w = (patch_size - orig_W % patch_size) % patch_size
            
            if pad_h > 0 or pad_w > 0:
                img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                
            # Extract features
            crop_emb = embed_single(img_tensor, model, patch_size=patch_size, embed_mode=embed_mode).cpu()
            
            # Crop the embeddings back to the original requested dimensions
            if embed_mode == "pixel" and (pad_h > 0 or pad_w > 0):
                crop_emb = crop_emb[:, :, :orig_H, :orig_W]
                
            # Store output without the batch dimension: Shape (D, H, W)
            extracted_features[volume_name][crop_id] = crop_emb.squeeze(0)

    print(f"\nFeature embedding complete ...")

    return extracted_features