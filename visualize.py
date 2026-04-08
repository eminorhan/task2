# import zarr
# import json
# import matplotlib.pyplot as plt
# from pathlib import Path


# def plot_retrieval_results(
#     sim_maps,
#     selected_queries_dict, 
#     crop_config,
#     base_data_dir,
#     output_filename="retrieval_results.jpeg",
# ):
#     """
#     Plots a 3x3 grid of raw EM crops with overlaid similarity heatmaps and query locations.
#     Saves the final plot directly to a high-resolution JPEG file.
#     """
#     # Get the valid globally unique tuples to filter the red dots
#     active_query_tuples = list(selected_queries_dict.keys())


#     volumes = list(crop_config.keys())
#     if len(volumes) != 3:
#         print("Warning: Expected exactly 3 volumes for a 3x3 grid.")

#     # Setup the 3x3 figure with ZERO whitespace internally
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
    
#     for row_idx, volume_name in enumerate(volumes):
#         crops = crop_config[volume_name]
        
#         # Open the Zarr array for this volume
#         em_dir = Path(base_data_dir) / volume_name / f"{volume_name}.zarr" / "recon-1" / "em"
#         modality_dirs = list(em_dir.glob("fibsem-*")) + list(em_dir.glob("tem-*"))
        
#         if not modality_dirs:
#             print(f"Warning: No imaging modality folder found for {volume_name}. Skipping.")
#             continue
            
#         dataset = zarr.open(str(modality_dirs[0] / "s0"), mode='r')

#         for col_idx, crop in enumerate(crops):
#             if col_idx >= 3:
#                 break
                
#             ax = axes[row_idx, col_idx]
#             crop_id = crop["crop_id"]
            
#             # 1. Fetch Raw Image
#             z = crop["z"]
#             y1, y2, x1, x2 = crop["y_min"], crop["y_max"], crop["x_min"], crop["x_max"]
#             raw_img = dataset[z, y1:y2, x1:x2]
            
#             # 2. Fetch Computed Similarity Map
#             sim_map_upscaled = sim_maps[volume_name][crop_id]
            
#             # 3. Plotting
#             ax.imshow(raw_img, cmap='gray')
#             im = ax.imshow(sim_map_upscaled, cmap='viridis', alpha=0.4, vmin=0.0, vmax=1.0)
            
#             # 4. Plot Query Locations
#             if "queries" in crop:
#                 for q in crop["queries"]:
#                     # Create the current tuple to check against our active list
#                     current_tuple = (volume_name, crop_id, q["query_id"])
                    
#                     # ONLY draw the dot if the Volume + Crop + ID match exactly
#                     if current_tuple in active_query_tuples:
#                         local_x = q["x_center"] - x1
#                         local_y = q["y_center"] - y1
#                         ax.plot(local_x, local_y, marker='o', color='red', markersize=5, markeredgecolor='white', markeredgewidth=1)
                        
#             # Add Z-index text to the top right of the plot
#             ax.text(
#                 0.99, 0.99, f"z = {z}", 
#                 color='white', fontsize=9, fontweight='bold',
#                 ha='right', va='top', transform=ax.transAxes
#             )
            
#             # Turn off standard tick marks and borders, but DO NOT use axis('off') 
#             ax.set_xticks([])
#             ax.set_yticks([])
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
                
#             # Add Volume Name as Y-Label (Only for the first column)
#             if col_idx == 0:
#                 ax.set_ylabel(volume_name, fontsize=12, fontweight='bold')
                
#             ax.set_aspect('auto')
            
#         # Clean up empty subplots (safe to use axis('off') here since they are blank)
#         for col_idx in range(len(crops), 3):
#             axes[row_idx, col_idx].axis('off')

#     plt.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.close(fig) 
#     print(f"Successfully saved visualization to {output_filename}")

from pathlib import Path
import matplotlib.pyplot as plt
import zarr

def plot_retrieval_results(
    sim_maps,
    selected_queries_dict, 
    crop_config,
    base_data_dir,
    output_filename="retrieval_results.jpeg",
    plot_raw=False,
):
    """
    Plots a 3x3 grid of raw EM crops with overlaid similarity heatmaps and active query locations.
    Saves the final plot directly to a high-resolution JPEG file.
    If plot_raw is True, additionally generates and saves a separate JPEG file 
    containing the raw EM images with ALL query locations plotted.
    """
    # Get the valid globally unique tuples to filter the red dots for the overlaid plot
    active_query_tuples = list(selected_queries_dict.keys())

    volumes = list(crop_config.keys())
    if len(volumes) != 3:
        print("Warning: Expected exactly 3 volumes for a 3x3 grid.")

    # Setup the 3x3 figure with ZERO whitespace internally for overlaid plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
    
    # Setup parallel figure for raw images if requested
    if plot_raw:
        fig_raw, axes_raw = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
    
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
            if plot_raw:
                ax_raw = axes_raw[row_idx, col_idx]
                
            crop_id = crop["crop_id"]
            
            # 1. Fetch Raw Image
            z = crop["z"]
            y1, y2, x1, x2 = crop["y_min"], crop["y_max"], crop["x_min"], crop["x_max"]
            raw_img = dataset[z, y1:y2, x1:x2]
            
            # 2. Fetch Computed Similarity Map
            sim_map_upscaled = sim_maps[volume_name][crop_id]
            
            # 3. Plotting Base Image
            ax.imshow(raw_img, cmap='gray')
            if plot_raw:
                ax_raw.imshow(raw_img, cmap='gray')
            
            # Plot Overlays (Only on the primary figure)
            ax.imshow(sim_map_upscaled, cmap='viridis', alpha=0.4, vmin=0.0, vmax=1.0)
            
            # 4. Plot Query Locations
            if "queries" in crop:
                for q in crop["queries"]:
                    local_x = q["x_center"] - x1
                    local_y = q["y_center"] - y1
                    
                    # Plot ALL queries on the raw image (if enabled)
                    if plot_raw:
                        ax_raw.plot(local_x, local_y, marker='o', color='red', markersize=5, markeredgecolor='white', markeredgewidth=1)
                    
                    # Create the current tuple to check against our active list
                    current_tuple = (volume_name, crop_id, q["query_id"])
                    
                    # ONLY draw the dot on the overlaid image if the Volume + Crop + ID match exactly
                    if current_tuple in active_query_tuples:
                        ax.plot(local_x, local_y, marker='o', color='red', markersize=5, markeredgecolor='white', markeredgewidth=1)
                        
            # Format aesthetics for both axes if plot_raw is True
            target_axes = [ax, ax_raw] if plot_raw else [ax]
            
            for current_ax in target_axes:
                # Add Z-index text to the top right of the plot
                current_ax.text(
                    0.99, 0.99, f"z = {z}", 
                    color='white', fontsize=9, fontweight='bold',
                    ha='right', va='top', transform=current_ax.transAxes
                )
                
                # Turn off standard tick marks and borders
                current_ax.set_xticks([])
                current_ax.set_yticks([])
                for spine in current_ax.spines.values():
                    spine.set_visible(False)
                    
                # Add Volume Name as Y-Label (Only for the first column)
                if col_idx == 0:
                    current_ax.set_ylabel(volume_name, fontsize=12, fontweight='bold')
                    
                current_ax.set_aspect('auto')
            
        # Clean up empty subplots (safe to use axis('off') here since they are blank)
        for col_idx in range(len(crops), 3):
            axes[row_idx, col_idx].axis('off')
            if plot_raw:
                axes_raw[row_idx, col_idx].axis('off')

    # Save Overlaid Results
    fig.savefig(output_filename, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 
    print(f"Successfully saved overlaid visualization to {output_filename}")
    
    # Save Raw Results
    if plot_raw:
        raw_output_filename = "visuals/images.jpeg"
        
        fig_raw.savefig(raw_output_filename, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig_raw)
        print(f"Successfully saved raw image grid to {raw_output_filename}")