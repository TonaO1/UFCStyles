"""
Server-side numbers for one k6 run, from CloudWatch.

    python loadtest/cloudwatch_metrics.py loadtest/results/<run>.json

Reads the run's start/end from the k6 result file, the API id and Lambda name from
`terraform -chdir=infra output`, and pulls for that window (padded by a minute each side):

  API Gateway  Count, 4xx, 5xx, Latency and IntegrationLatency percentiles
  Lambda       Invocations, Errors, Throttles, Duration percentiles, peak ConcurrentExecutions
  Lambda logs  cold starts and init duration, from REPORT lines (Logs Insights)

CloudWatch metrics land 1-3 minutes late, so run this a few minutes after k6 finishes.
Writes <run>.cloudwatch.json next to the k6 file.
"""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

PAD = timedelta(minutes=1)


def terraform_outputs() -> dict:
    out = subprocess.run(["terraform", "-chdir=infra", "output", "-json"], check=True, capture_output=True, text=True)
    return {k: v["value"] for k, v in json.loads(out.stdout).items()}


def metric_queries(api_id: str, function: str) -> list[dict]:
    def q(qid, namespace, name, dims, stat):
        return {"Id": qid, "Label": f"{namespace} {name} {stat}",
                "MetricStat": {"Metric": {"Namespace": namespace, "MetricName": name,
                                          "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()]},
                               "Stat": stat}}

    api, fn = {"ApiId": api_id}, {"FunctionName": function}
    queries = [
        q("api_count", "AWS/ApiGateway", "Count", api, "Sum"),
        q("api_4xx", "AWS/ApiGateway", "4xx", api, "Sum"),
        q("api_5xx", "AWS/ApiGateway", "5xx", api, "Sum"),
        q("lambda_invocations", "AWS/Lambda", "Invocations", fn, "Sum"),
        q("lambda_errors", "AWS/Lambda", "Errors", fn, "Sum"),
        q("lambda_throttles", "AWS/Lambda", "Throttles", fn, "Sum"),
        q("lambda_concurrency_max", "AWS/Lambda", "ConcurrentExecutions", fn, "Maximum"),
        q("lambda_duration_max", "AWS/Lambda", "Duration", fn, "Maximum"),
    ]
    for p in ("p50", "p95", "p99"):
        queries += [
            q(f"api_latency_{p}", "AWS/ApiGateway", "Latency", api, p),
            q(f"api_integration_latency_{p}", "AWS/ApiGateway", "IntegrationLatency", api, p),
            q(f"lambda_duration_{p}", "AWS/Lambda", "Duration", fn, p),
        ]
    return queries


def get_metrics(cw, queries, start, end) -> dict:
    # One period covering the whole window, so each query returns a single aggregate.
    period = max(60, math.ceil((end - start).total_seconds() / 60) * 60)
    for query in queries:
        query["MetricStat"]["Period"] = period
    res = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=start + timedelta(seconds=period),
                             ScanBy="TimestampAscending")
    out = {}
    for r in res["MetricDataResults"]:
        vals = r["Values"]
        # Sums over a single period; Maximum/percentiles take the single value.
        out[r["Id"]] = round(vals[0], 2) if vals else None
    return out


def get_cold_starts(logs, log_group, start, end) -> dict:
    query = ('filter @type = "REPORT" '
             '| stats count(*) as invocations, count(@initDuration) as cold_starts, '
             'avg(@initDuration) as init_avg_ms, max(@initDuration) as init_max_ms, '
             'pct(@duration, 50) as duration_p50_ms, pct(@duration, 95) as duration_p95_ms, '
             'pct(@duration, 99) as duration_p99_ms, max(@maxMemoryUsed) / 1000000 as max_memory_mb')
    qid = logs.start_query(logGroupName=log_group, startTime=int(start.timestamp()), endTime=int(end.timestamp()),
                           queryString=query)["queryId"]
    while True:
        res = logs.get_query_results(queryId=qid)
        if res["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        time.sleep(1)
    if res["status"] != "Complete" or not res["results"]:
        return {"status": res["status"]}
    return {f["field"]: round(float(f["value"]), 2) for f in res["results"][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="k6 result file from loadtest/results/")
    ap.add_argument("--region", default="us-east-2")
    args = ap.parse_args()

    run = json.loads(args.run.read_text())
    start = datetime.fromisoformat(run["start"].replace("Z", "+00:00")).replace(second=0, microsecond=0) - PAD
    end = datetime.fromisoformat(run["end"].replace("Z", "+00:00")) + PAD
    if datetime.now(timezone.utc) - end < timedelta(minutes=2):
        print("warning: the run ended under 3 minutes ago; CloudWatch may not have all of it yet")

    tf = terraform_outputs()
    api_id = tf["api_url"].split("//")[1].split(".")[0]
    function = tf["lambda_function_name"]

    session = boto3.Session(region_name=args.region)
    metrics = get_metrics(session.client("cloudwatch"), metric_queries(api_id, function), start, end)
    cold = get_cold_starts(session.client("logs"), f"/aws/lambda/{function}", start, end)

    result = {"run": args.run.name, "window": [start.isoformat(), end.isoformat()],
              "api_id": api_id, "function": function, "metrics": metrics, "lambda_reports": cold}
    out = args.run.with_suffix(".cloudwatch.json")
    out.write_text(json.dumps(result, indent=2))

    m = metrics
    print(f"\nCloudWatch, {start:%H:%M}-{end:%H:%M} UTC ({args.run.name})")
    print(f"  API Gateway  requests {m['api_count']}  4xx {m['api_4xx']}  5xx {m['api_5xx']}")
    print(f"               latency p50/p95/p99 {m['api_latency_p50']} / {m['api_latency_p95']} / {m['api_latency_p99']} ms")
    print(f"               integration p50/p95/p99 {m['api_integration_latency_p50']} / "
          f"{m['api_integration_latency_p95']} / {m['api_integration_latency_p99']} ms")
    print(f"  Lambda       invocations {m['lambda_invocations']}  errors {m['lambda_errors']}  "
          f"throttles {m['lambda_throttles']}  peak concurrency {m['lambda_concurrency_max']}")
    print(f"               duration p50/p95/p99/max {m['lambda_duration_p50']} / {m['lambda_duration_p95']} / "
          f"{m['lambda_duration_p99']} / {m['lambda_duration_max']} ms")
    print(f"  Cold starts  {cold.get('cold_starts')} of {cold.get('invocations')}  "
          f"init avg {cold.get('init_avg_ms')} ms  max {cold.get('init_max_ms')} ms  "
          f"memory max {cold.get('max_memory_mb')} MB")
    print(f"  saved        {out}\n")


if __name__ == "__main__":
    main()
