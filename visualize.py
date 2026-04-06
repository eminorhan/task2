import zarr
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from retrieve import compute_similarity


def select_queries(query_dict, *selections):
    """
    Extracts queries and assigns them a globally unique tuple key: (volume, crop, query_id).
    Accepts multiple selections as tuples: (volume, crop) or (volume, crop, q_ids).
    If q_ids is not provided or is None, it extracts all queries for that crop.
    """
    selected_queries = {}
    
    for selection in selections:
        volume = selection[0]
        crop = selection[1]
        q_ids = selection[2] if len(selection) > 2 else None
        
        crop_queries = query_dict[volume][crop]
        if q_ids is None:
            q_ids = list(crop_queries.keys()) # Grab all available IDs
            
        for q_id in q_ids:
            selected_queries[(volume, crop, q_id)] = crop_queries[q_id]
            
    return selected_queries


def plot_retrieval_results(
    selected_queries_dict, 
    embeddings_dict, 
    crop_config,
    base_data_dir,
    method="average",
    output_filename="retrieval_results.jpeg"
):
    """
    Plots a 3x3 grid of raw EM crops with overlaid similarity heatmaps and query locations.
    Saves the final plot directly to a high-resolution JPEG file.
    """
    # Stack all tensors in the dict into an (N, D) tensor for similarity computation
    query_tensor = torch.stack(list(selected_queries_dict.values()))
    
    # Get the valid globally unique tuples to filter the red dots
    active_query_tuples = list(selected_queries_dict.keys())

    volumes = list(crop_config.keys())
    if len(volumes) != 3:
        print("Warning: Expected exactly 3 volumes for a 3x3 grid.")

    # Setup the 3x3 figure with ZERO whitespace internally
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
    
    for row_idx, volume_name in enumerate(volumes):
        crops = crop_config[volume_name]
        
        # Open the Zarr array for this volume
        em_dir = Path(base_data_dir) / volume_name / f"{volume_name}.zarr" / "recon-1" / "em"
        modality_dirs = list(em_dir.glob("fibsem-*")) + list(em_dir.glob("tem-*"))
        
        if not modality_dirs:
            print(f"Warning: No imaging modality folder found for {volume_name}. Skipping.")
            continue
            
        dataset = zarr.open(str(modality_dirs[0] / "s0"), mode='r')

        for col_idx, crop in enumerate(crops):
            if col_idx >= 3:
                break
                
            ax = axes[row_idx, col_idx]
            crop_id = crop["crop_id"]
            
            # 1. Fetch Raw Image
            z = crop["z"]
            y1, y2, x1, x2 = crop["y_min"], crop["y_max"], crop["x_min"], crop["x_max"]
            raw_img = dataset[z, y1:y2, x1:x2]
            
            # 2. Fetch Embeddings & Compute Similarity
            target_emb = embeddings_dict[volume_name][crop_id]
            sim_map = compute_similarity(query_tensor, target_emb, method=method)
            
            # Upscale the heatmap to match the raw image resolution
            sim_map = sim_map.unsqueeze(0).unsqueeze(0) 
            sim_map_upscaled = F.interpolate(
                sim_map, 
                size=(raw_img.shape[0], raw_img.shape[1]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # 3. Plotting
            ax.imshow(raw_img, cmap='gray')
            im = ax.imshow(sim_map_upscaled, cmap='viridis', alpha=0.4, vmin=0.0, vmax=1.0)
            
            # 4. Plot Query Locations
            if "queries" in crop:
                for q in crop["queries"]:
                    # Create the current tuple to check against our active list
                    current_tuple = (volume_name, crop_id, q["query_id"])
                    
                    # ONLY draw the dot if the Volume + Crop + ID match exactly
                    if current_tuple in active_query_tuples:
                        local_x = q["x_center"] - x1
                        local_y = q["y_center"] - y1
                        ax.plot(local_x, local_y, marker='o', color='red', markersize=5, markeredgecolor='white', markeredgewidth=1)
                        
            # Add Z-index text to the top right of the plot
            ax.text(
                0.99, 0.99, f"z = {z}", 
                color='white', fontsize=9, fontweight='bold',
                ha='right', va='top', transform=ax.transAxes
            )
            
            # Turn off standard tick marks and borders, but DO NOT use axis('off') 
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
                
            # Add Volume Name as Y-Label (Only for the first column)
            if col_idx == 0:
                ax.set_ylabel(volume_name, fontsize=12, fontweight='bold')
                
            ax.set_aspect('auto')
            
        # Clean up empty subplots (safe to use axis('off') here since they are blank)
        for col_idx in range(len(crops), 3):
            axes[row_idx, col_idx].axis('off')

    plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 
    print(f"Successfully saved {method} visualization to {output_filename}")