import torch
import torch.nn.functional as F


def extract_queries(embeddings_dict, crop_config, embed_mode="pixel", patch_size=16):
    """
    Iterates through the JSON config, translates coordinates, 
    and extracts all query vectors into a structured nested dictionary.
    
    Returns:
        dict: { volume_name: { crop_id: { query_id: tensor(D,) } } }
    """
    all_queries = {}

    for volume_name, crops in crop_config.items():
        if volume_name not in embeddings_dict:
            print(f"Warning: {volume_name} not found in pre-computed embeddings. Skipping.")
            continue
            
        all_queries[volume_name] = {}
        
        for crop in crops:
            crop_id = crop["crop_id"]
            if crop_id not in embeddings_dict[volume_name]:
                continue
                
            if "queries" not in crop or not crop["queries"]:
                continue # Skip crops that are only used as targets, not queries
                
            all_queries[volume_name][crop_id] = {}
            crop_emb = embeddings_dict[volume_name][crop_id] # Shape: (D, H, W)
            
            y_min, x_min = crop["y_min"], crop["x_min"]
            
            for q in crop["queries"]:
                query_id = q["query_id"]
                
                # Step A: Global -> Local Crop Space
                y_local = q["y_center"] - y_min
                x_local = q["x_center"] - x_min
                
                # Step B: Local Crop -> Embedding Space
                if embed_mode == "patch":
                    y_emb = y_local // patch_size
                    x_emb = x_local // patch_size
                else: # "pixel" mode
                    y_emb = y_local
                    x_emb = x_local
                    
                # Extract the 1D embedding vector: Shape (D,)
                vec = crop_emb[:, y_emb, x_emb]
                all_queries[volume_name][crop_id][query_id] = vec
                
    # Quick summary printout
    total_queries = sum(len(queries) for vols in all_queries.values() for queries in vols.values())
    print(f"Successfully extracted {total_queries} distinct query vectors.")
    
    return all_queries


def compute_similarity(query_vectors, target_embeddings, method="maxsim"):
    """
    Computes cosine similarity between queries and target embeddings.
    
    Args:
        query_vectors (torch.Tensor): Shape (D,) or (N, D).
        target_embeddings (torch.Tensor): Shape (D, H, W).
        method (str): 'average' or 'maxsim'.
        
    Returns:
        torch.Tensor: Shape (H, W) containing similarity scores.
    """
    device = target_embeddings.device
    query_vectors = query_vectors.to(device)
    
    # Standardize to (N, D)
    if query_vectors.dim() == 1:
        query_vectors = query_vectors.unsqueeze(0)
        
    D, H, W = target_embeddings.shape
    N = query_vectors.shape[0]

    if method == "average":
        # Average queries first, then compute similarity
        q_avg = query_vectors.mean(dim=0).view(D, 1, 1)
        sim_map = F.cosine_similarity(q_avg, target_embeddings, dim=0)
        
    elif method == "maxsim":
        # Compute all similarities, then take the max at each pixel
        all_sims = []
        for i in range(N):
            q = query_vectors[i].view(D, 1, 1)
            sim = F.cosine_similarity(q, target_embeddings, dim=0)
            all_sims.append(sim)
            
        # Stack to (N, H, W) and take max along the N dimension
        sim_map = torch.stack(all_sims).max(dim=0)[0]
        
    else:
        raise ValueError("Method must be 'average' or 'maxsim'")
        
    return sim_map    