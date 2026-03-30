import os
import argparse
import json
import torch

from embed import embed_all 
from retrieve import extract_queries
from visualize import plot_retrieval_results, select_queries

# Backbone name -> file name mapping (you can add a few more DINOv3 checkpoints below)
BACKBONE_DICT = {
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    # "dinov3_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="End to end DINOv3 retrieval pipeline")
    
    # --- Data & config arguments ---
    parser.add_argument("--config", type=str, default="selected_crops.json", help="Path to JSON config.")
    parser.add_argument("--data_dir", type=str, default="data", help="Base directory containing EM volumes.")
    
    # --- Embedding arguments ---
    parser.add_argument("--dinov3_repo_path", type=str, default="../dinov3", help="Local DINOv3 repo path.")
    parser.add_argument("--torch_hub_path", type=str, default="../torch_hub", help="Local Torch Hub path (where the checkpoints are stored).")
    parser.add_argument("--backbone", type=str, default="dinov3_vith16plus", help="Name of the DINOv3 backbone.")
    parser.add_argument("--embed_mode", type=str, choices=["pixel", "patch"], default="patch", help="Embedding mode (per pixel or per patch).")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size of the model (16 for all DINOv3 backbones).")
        
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    os.makedirs("visuals", exist_ok=True)  # Create a "visuals" directory for dumping images
    
    # Full path where the pretrained .pth checkpoint is stored
    weights_path = os.path.join(args.torch_hub_path, "checkpoints", BACKBONE_DICT[args.backbone])

    # Set torch_hub dir
    torch.hub.set_dir(args.torch_hub_path)

    # Load the model
    model = torch.hub.load(
        args.dinov3_repo_path, 
        args.backbone, 
        source="local", 
        weights=weights_path, 
        pretrained=True, 
        use_fa3=False
    )
    model = model.to(device)
    model.eval()

    # Load the crop config file containing the selected crops and query locations
    with open(args.config, 'r') as f:
        crop_config = json.load(f)
    
    # Embed all selected slices
    embeddings_dict = embed_all(
        crop_config,
        model,
        data_dir=args.data_dir,
        embed_mode=args.embed_mode,
        patch_size=args.patch_size,
        device=device
    )

    # Extract all queries 
    query_dict = extract_queries(embeddings_dict, crop_config, embed_mode=args.embed_mode, patch_size=args.patch_size)

    # === Single query example ===
    single_query = select_queries(query_dict, "jrc_jurkat-1", "jurkat_1_1", ["q1"])

    plot_retrieval_results(
        selected_queries_dict=single_query,
        embeddings_dict=embeddings_dict,
        crop_config=crop_config,
        base_data_dir=args.data_dir,
        output_filename=f"visuals/single_query_{args.backbone}_{args.embed_mode}.jpeg"
    )
    # ===================================

    # === Multi-query average example ===
    multi_query_avg = select_queries(query_dict, "jrc_jurkat-1", "jurkat_1_1", ["q1", "q2", "q3"])

    plot_retrieval_results(
        selected_queries_dict=multi_query_avg,
        embeddings_dict=embeddings_dict,
        crop_config=crop_config,
        base_data_dir=args.data_dir,
        method="average",
        output_filename=f"visuals/multi_query_avg_{args.backbone}_{args.embed_mode}.jpeg"
    )
    # ===================================

    # === Multi-query maxsim example ===
    multi_query_maxsim = select_queries(query_dict, "jrc_jurkat-1", "jurkat_1_1", ["q1", "q4", "q6"])

    plot_retrieval_results(
        selected_queries_dict=multi_query_maxsim,
        embeddings_dict=embeddings_dict,
        crop_config=crop_config,
        base_data_dir=args.data_dir,
        method="maxsim",
        output_filename=f"visuals/multi_query_maxsim_{args.backbone}_{args.embed_mode}.jpeg"
    )
    # ===================================
