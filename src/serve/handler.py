"""
AWS Lambda handler behind an API Gateway HTTP API.

  GET /similar?fighter=<name or id>&k=10  -> closest fighters by style embedding
  GET /matchup?a=<name or id>&b=<name or id> -> P(a beats b) from the harness fight model

Fighter records live in DynamoDB (src/serve/load_dynamodb.py fills it). fight_model.npz
ships next to this file. Env vars: TABLE_NAME, FIGHT_MODEL_PATH.
"""

import json
import os
import sys
import traceback
from decimal import Decimal
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_npz, most_similar, p_a_wins

TABLE_NAME = os.environ.get("TABLE_NAME", "ufc-fighter-embeddings-dev")
FIGHT_MODEL_PATH = os.environ.get("FIGHT_MODEL_PATH", str(Path(__file__).resolve().parent / "fight_model.npz"))
MAX_K = 50

# Survives between requests while the Lambda stays warm.
_cache = {}


class ClientError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _from_dynamo(value):
    """DynamoDB hands every number back as Decimal; turn them into floats."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_from_dynamo(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_dynamo(v) for k, v in value.items()}
    return value


def _table():
    if "table" not in _cache:
        _cache["table"] = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _cache["table"]


def _fighters() -> dict:
    """Scan the table once per warm Lambda. ~1,000 small records fit in memory easily."""
    if "fighters" not in _cache:
        items, kwargs = [], {}
        while True:
            page = _table().scan(**kwargs)
            items += page["Items"]
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        records = [_from_dynamo(item) for item in items]
        by_name = {}
        for n, r in enumerate(records):
            by_name.setdefault(r["fighter_name"].lower(), []).append(n)
        _cache["fighters"] = {
            "records": records,
            "Z": np.array([r["embedding"] for r in records]),
            "by_id": {r["fighter_id"]: n for n, r in enumerate(records)},
            "by_name": by_name,
        }
    return _cache["fighters"]


def _fight_model() -> dict:
    if "fight_model" not in _cache:
        _cache["fight_model"] = load_npz(FIGHT_MODEL_PATH)
    return _cache["fight_model"]


def _find(key: str) -> int:
    f = _fighters()
    if key in f["by_id"]:
        return f["by_id"][key]
    matches = f["by_name"].get(key.strip().lower(), [])
    if not matches:
        raise ClientError(404, f"fighter not found: {key}")
    if len(matches) > 1:
        ids = [f["records"][n]["fighter_id"] for n in matches]
        raise ClientError(409, f"{len(matches)} fighters are named {key}; use an id: {ids}")
    return matches[0]


def _summary(record: dict) -> dict:
    return {k: record.get(k) for k in ("fighter_id", "fighter_name", "weight_class", "background", "snapshot_date")}


def get_similar(fighter: str, k: int) -> dict:
    f = _fighters()
    i = _find(fighter)
    top, sims = most_similar(f["Z"], i, k)
    return {"fighter": _summary(f["records"][i]),
            "similar": [{**_summary(f["records"][t]), "similarity": round(float(s), 4)} for t, s in zip(top, sims)]}


def get_matchup(a: str, b: str) -> dict:
    f = _fighters()
    ra, rb = f["records"][_find(a)], f["records"][_find(b)]
    if ra["fighter_id"] == rb["fighter_id"]:
        raise ClientError(400, "a and b are the same fighter")
    fm = _fight_model()
    p = p_a_wins(fm, ra["strength"], rb["strength"], np.array(ra["embedding"]), np.array(rb["embedding"]))
    return {"a": _summary(ra), "b": _summary(rb), "p_a_wins": round(p, 4), "p_b_wins": round(1 - p, 4),
            "style_term_used": bool(np.any(fm["W"]))}


def _required(params: dict, name: str) -> str:
    value = params.get(name)
    if not value:
        raise ClientError(400, f"missing query parameter: {name}")
    return value


def lambda_handler(event, context):
    """Route an HTTP API (payload v2) request and always answer with JSON."""
    params = event.get("queryStringParameters") or {}
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    try:
        if method != "GET":
            raise ClientError(405, f"method not allowed: {method}")
        if path.endswith("/similar"):
            try:
                k = int(params.get("k", 10))
            except ValueError:
                raise ClientError(400, "k must be an integer")
            status, body = 200, get_similar(_required(params, "fighter"), min(max(k, 1), MAX_K))
        elif path.endswith("/matchup"):
            status, body = 200, get_matchup(_required(params, "a"), _required(params, "b"))
        else:
            raise ClientError(404, f"unknown route: {path}")
    except ClientError as e:
        status, body = e.status, {"error": str(e)}
    except Exception:
        traceback.print_exc()   # lands in CloudWatch Logs
        status, body = 500, {"error": "internal error"}

    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}
