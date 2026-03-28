import os
import argparse
import zarr
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ==========================================
# 1. Data Loading 
# ==========================================

def load_uniformly_spaced_slices(base_data_dir, volume_name, n_slices):
    """Reads uniformly spaced Z-slices from an OpenOrganelle Zarr structure."""
    
    # Navigate to the 'em' directory
    em_dir = Path(base_data_dir) / volume_name / f"{volume_name}.zarr" / "recon-1" / "em"
    
    # Dynamically find the fibsem directory (handles uint8, uint16, etc.)
    fibsem_dirs = list(em_dir.glob("fibsem-*"))
    
    if not fibsem_dirs:
        raise FileNotFoundError(f"Could not find any 'fibsem-*' directory in {em_dir}")
        
    # Take the first matched directory and append 's0'
    zarr_path = fibsem_dirs[0] / "s0"
    
    dataset = zarr.open(str(zarr_path), mode='r')
    
    total_z_slices = dataset.shape[0]
    z_indices = np.linspace(0, total_z_slices - 1, n_slices, dtype=int)
    
    print(f"Loading {n_slices} slices from {volume_name} ({fibsem_dirs[0].name})...")
    slices = [dataset[idx, :, :] for idx in z_indices]
    
    return slices, z_indices
    
# ==========================================
# 2. Batched model forward
# ==========================================

def extract_tile_embeddings(tile_tensor, model, patch_size=14, mode="pixel"):
    """
    Passes a BATCH of image tiles through DINOv3.
    tile_tensor: shape (B, 3, H, W)
    """
    B, _, H, W = tile_tensor.shape
    
    with torch.no_grad():
        outputs = model.forward_features(tile_tensor)
        patch_tokens = outputs["x_norm_patchtokens"] # Shape: (B, N, Embedding_Dim)
        embedding_dim = patch_tokens.shape[-1]
        
        h_patches, w_patches = H // patch_size, W // patch_size
        
        # Keep the B dimension dynamically sizing
        spatial_embeddings = patch_tokens.reshape(B, h_patches, w_patches, embedding_dim)
        spatial_embeddings = spatial_embeddings.permute(0, 3, 1, 2) # Shape: (B, D, H_p, W_p)
        
        if mode == "patch":
            return spatial_embeddings
        elif mode == "pixel":
            dense_embeddings = F.interpolate(spatial_embeddings, size=(H, W), mode="bilinear", align_corners=False)
            return dense_embeddings
        else:
            raise ValueError("Mode must be 'patch' or 'pixel'")

# ==========================================
# 3. Sliding Window Pipeline
# ==========================================

def process_volume_slices(slices, model, tile_size=512, stride=256, batch_size=8, mode="pixel", device="cuda"):
    """Processes EM slices using batched sliding window inference."""
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    all_slice_embeddings = []
    
    for idx, img_array in enumerate(slices):
        print(f"  -> Processing slice {idx+1}/{len(slices)}...")
        
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1) 
        img_tensor = normalize(img_tensor).unsqueeze(0).to(device) 
        
        _, _, H, W = img_tensor.shape
        scale_factor = 1 if mode == "pixel" else 14
        out_H, out_W = H // scale_factor, W // scale_factor
        
        # Dummy pass to get embedding dimension
        dummy_out = extract_tile_embeddings(torch.zeros(1, 3, tile_size, tile_size).to(device), model, mode=mode)
        embed_dim = dummy_out.shape[1]
        
        final_embeddings = torch.zeros((1, embed_dim, out_H, out_W), device='cpu')
        count_map = torch.zeros((1, 1, out_H, out_W), device='cpu')
        
        # Lists to hold our current batch
        batch_tiles = []
        batch_coords = []
        
        # Generate all coordinates first to easily know when we hit the last tile
        coords = [(y, x) for y in range(0, H - tile_size + stride, stride) 
                         for x in range(0, W - tile_size + stride, stride)]
        
        for i, (y, x) in enumerate(coords):
            y1, x1 = min(y, H - tile_size), min(x, W - tile_size)
            y2, x2 = y1 + tile_size, x1 + tile_size
            
            # Extract tile and append to batch list
            tile = img_tensor[:, :, y1:y2, x1:x2]
            batch_tiles.append(tile)
            
            # Save the target coordinates for later
            out_y1, out_y2 = y1 // scale_factor, y2 // scale_factor
            out_x1, out_x2 = x1 // scale_factor, x2 // scale_factor
            batch_coords.append((out_y1, out_y2, out_x1, out_x2))
            
            # If batch is full, OR it's the very last tile in the slice
            if len(batch_tiles) == batch_size or i == len(coords) - 1:
                # 1. Stack tiles along the batch dimension: (B, 3, H, W)
                stacked_tiles = torch.cat(batch_tiles, dim=0)
                
                # 2. Run the batched forward pass
                batch_emb = extract_tile_embeddings(stacked_tiles, model, mode=mode).cpu()
                
                # 3. Unpack the batch and add to the final map
                for b in range(len(batch_tiles)):
                    oy1, oy2, ox1, ox2 = batch_coords[b]
                    # Keep the 1-dim batch shape for broadcasting: [1, D, H, W]
                    final_embeddings[:, :, oy1:oy2, ox1:ox2] += batch_emb[b:b+1]
                    count_map[:, :, oy1:oy2, ox1:ox2] += 1
                
                # 4. Clear the batch lists for the next round
                batch_tiles = []
                batch_coords = []

        final_embeddings = final_embeddings / count_map
        all_slice_embeddings.append(final_embeddings.squeeze(0)) 
        
    return all_slice_embeddings

# ==========================================
# 4. Main Execution Loop
# ==========================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print(f"Loading model: {args.backbone}...")
    
    model = torch.hub.load(
        args.repo_path, 
        args.backbone, 
        source="local", 
        weights=args.weights_path, 
        pretrained=True, 
        use_fa3=False
    )
    model = model.to(device)
    model.eval() 
    
    all_volume_embeddings = {}

    for volume_name in args.volumes:
        print(f"\n--- Processing Volume: {volume_name} ---")
        try:
            slices, z_indices = load_uniformly_spaced_slices(args.data_dir, volume_name, args.n_slices)
        except Exception as e:
            print(f"Failed to load {volume_name}: {e}")
            continue
            
        print(f"Extracting {args.mode} embeddings...")
        slice_embeddings = process_volume_slices(
            slices, model, 
            tile_size=args.tile_size, 
            stride=args.stride, 
            mode=args.mode, 
            device=device
        )
        
        all_volume_embeddings[volume_name] = {"z_indices": z_indices, "embeddings": slice_embeddings}
        
    print(f"\nFeature extraction complete! Saving to {args.output}...")
    torch.save(all_volume_embeddings, args.output)
    print("Done.")

# ==========================================
# 5. CLI Configuration
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv3 embeddings from EM volumes.")
    
    # Data settings
    parser.add_argument("--data_dir", type=str, default="data", help="Base directory containing EM volumes.")
    parser.add_argument("--volumes", type=str, nargs="+", required=True, help="List of volume names to process (e.g., jrc_hela-1 jrc_macrophage-2).")
    parser.add_argument("--n_slices", type=int, default=5, help="Number of uniformly spaced z-slices to extract per volume.")
    
    # Model & Inference settings
    parser.add_argument("--repo_path", type=str, default="facebookresearch/dinov3", help="Path to DINOv3 repo.")
    parser.add_argument("--backbone", type=str, default="dinov3_vits14", help="Name of the DINOv3 backbone.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--mode", type=str, choices=["pixel", "patch"], default="pixel", help="Output resolution mode.")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size for sliding window inference.")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window inference.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of tiles to process simultaneously.")

    # Output settings
    parser.add_argument("--output", type=str, default="computed_embeddings.pt", help="Filepath to save the resulting PyTorch dictionary.")
    
    args = parser.parse_args()
    main(args)