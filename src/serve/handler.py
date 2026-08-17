"""
AWS Lambda handler for serving style embeddings.

Endpoint contracts:
  GET /similar?fighter=<name>&k=10  ->  {"fighter": name, "similar": [...]}
  GET /matchup?a=<name>&b=<name>    ->  {"a": name, "b": name, "p_a_wins": 0.65, ...}

You write:
  - NumPy-based encoder (no PyTorch in Lambda)
  - DynamoDB queries
  - Response formatting
"""

import json
import boto3
import numpy as np
from pathlib import Path

# ============================================================================
# INITIALIZATION
# ============================================================================

# Load encoder weights and scaler
ENCODER_WEIGHTS = np.load("encoder_weights.npz")
# Parse weights into layers
enc_layers = []
i = 0
while f"arr_{i}" in ENCODER_WEIGHTS:
    enc_layers.append(ENCODER_WEIGHTS[f"arr_{i}"])
    i += 1

SCALER_MEAN = np.array([0.0] * 28)  # Placeholder; load from scaler.pkl if needed
SCALER_SCALE = np.array([1.0] * 28)

# DynamoDB client
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("ufc_fighter_embeddings")


# ============================================================================
# ENCODER (NUMPY, NO PYTORCH)
# ============================================================================

def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def encode(x: np.ndarray) -> np.ndarray:
    """
    Encode features to embedding using NumPy.
    
    Args:
        x: (d_in,) feature vector
    
    Returns:
        (d_latent,) embedding
    """
    # Normalize
    x = (x - SCALER_MEAN) / (SCALER_SCALE + 1e-8)
    
    # Forward pass through encoder layers
    # Assuming encoder_weights has alternating W and b:
    # (W1, W2, W3, b1, b2, b3) or similar
    
    for i in range(0, len(enc_layers) - 1, 2):
        w = enc_layers[i]
        b = enc_layers[i + 1] if i + 1 < len(enc_layers) else None
        
        x = x @ w.T + (b if b is not None else 0)
        
        # GELU on all but the last layer
        if i + 2 < len(enc_layers):
            x = gelu(x)
    
    return x


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ============================================================================
# ENDPOINT: /similar
# ============================================================================

def get_similar_fighters(fighter_name: str, k: int = 10) -> dict:
    """
    Find k fighters most similar in embedding space.
    
    Args:
        fighter_name: fighter name (must be in DynamoDB)
        k: number of results
    
    Returns:
        dict with similar fighters and similarities
    """
    
    # Lookup fighter in DynamoDB
    response = table.get_item(Key={"fighter_id": fighter_name})
    
    if "Item" not in response:
        return {"error": f"Fighter '{fighter_name}' not found"}
    
    fighter_item = response["Item"]
    fighter_embedding = np.array(fighter_item["embedding"])
    
    # Scan all fighters and compute distances
    # (This is brute-force; at scale, use a vector DB)
    response = table.scan()
    
    similarities = []
    for item in response.get("Items", []):
        other_name = item.get("fighter_id")
        if other_name == fighter_name:
            continue
        
        other_embedding = np.array(item.get("embedding", []))
        if len(other_embedding) != len(fighter_embedding):
            continue
        
        sim = cosine_similarity(fighter_embedding, other_embedding)
        similarities.append((other_name, sim))
    
    # Top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    
    return {
        "fighter": fighter_name,
        "similar": [
            {"name": name, "similarity": float(sim)}
            for name, sim in top_k
        ]
    }


# ============================================================================
# ENDPOINT: /matchup
# ============================================================================

def get_matchup(fighter_a: str, fighter_b: str) -> dict:
    """
    Predict matchup outcome based on embedding difference.
    
    Args:
        fighter_a, fighter_b: fighter names
    
    Returns:
        dict with P(a wins), P(b wins), matchup notes
    """
    
    # Lookup both fighters
    resp_a = table.get_item(Key={"fighter_id": fighter_a})
    resp_b = table.get_item(Key={"fighter_id": fighter_b})
    
    if "Item" not in resp_a:
        return {"error": f"Fighter '{fighter_a}' not found"}
    if "Item" not in resp_b:
        return {"error": f"Fighter '{fighter_b}' not found"}
    
    emb_a = np.array(resp_a["Item"]["embedding"])
    emb_b = np.array(resp_b["Item"]["embedding"])
    
    # Simple model: P(A wins) = sigmoid(w @ (e_a - e_b))
    # For now, use random weight (you would train this)
    w = np.random.randn(len(emb_a))
    w = w / np.linalg.norm(w)
    
    logit = np.dot(w, emb_a - emb_b)
    p_a_wins = 1.0 / (1.0 + np.exp(-logit))
    p_b_wins = 1.0 - p_a_wins
    
    # Style differences (which dimensions differ most?)
    diff = emb_a - emb_b
    most_different = np.argsort(-np.abs(diff))[:3]
    
    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "p_a_wins": round(p_a_wins, 3),
        "p_b_wins": round(p_b_wins, 3),
        "style_diff_dimensions": most_different.tolist(),
        "note": "This is a style-based prediction, not a quality prediction. Results are for analysis only."
    }


# ============================================================================
# LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """
    API Gateway Lambda handler.
    
    Routes:
      - GET /similar?fighter=<name>&k=10
      - GET /matchup?a=<a_name>&b=<b_name>
    """
    
    try:
        http_method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
        path = event.get("rawPath", "")
        query_params = event.get("queryStringParameters", {})
        
        if path == "/similar" and http_method == "GET":
            fighter = query_params.get("fighter", "")
            k = int(query_params.get("k", 10))
            result = get_similar_fighters(fighter, k)
        
        elif path == "/matchup" and http_method == "GET":
            fighter_a = query_params.get("a", "")
            fighter_b = query_params.get("b", "")
            result = get_matchup(fighter_a, fighter_b)
        
        else:
            result = {"error": f"Unknown route: {path}"}
        
        return {
            "statusCode": 200 if "error" not in result else 404,
            "body": json.dumps(result),
            "headers": {"Content-Type": "application/json"}
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
