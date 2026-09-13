"""
Load a serving bundle's fighters.json into DynamoDB. Run after `terraform apply`.

  python src/serve/load_dynamodb.py --model contrastive_style_8 --table ufc-fighter-embeddings-dev
"""

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path

import boto3


def to_item(record: dict) -> dict:
    """DynamoDB rejects Python floats, so numbers go in as Decimal; empty fields are left out."""
    def convert(value):
        if isinstance(value, float):
            assert math.isfinite(value), f"non-finite number in {record['fighter_id']}"
            return Decimal(str(value))
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return {k: convert(v) for k, v in record.items() if v is not None}


def main(args):
    path = Path("data/models") / args.model / "serving" / "fighters.json"
    records = json.loads(path.read_text())
    table = boto3.resource("dynamodb", region_name=args.region).Table(args.table)

    with table.batch_writer(overwrite_by_pkeys=["fighter_id"]) as batch:
        for record in records:
            batch.put_item(Item=to_item(record))
    print(f"✓ wrote {len(records)} fighters from {path} to {args.table} ({args.region})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="contrastive_style_8")
    parser.add_argument("--table", required=True)
    parser.add_argument("--region", default="us-east-1")
    main(parser.parse_args())
