// k6 load test for the serving API (GET /similar, GET /matchup).
//
//   k6 run -e BASE_URL="$(terraform -chdir=infra output -raw api_url)" -e TEST=throttle loadtest/api.js
//   k6 run -e BASE_URL="$(terraform -chdir=infra output -raw api_url)" -e TEST=capacity loadtest/api.js
//
// TEST=throttle  constant RATE req/s (default 20) for DURATION (default 1m), against the
//                default 5 req/s stage limit. 429 counts as an expected response.
// TEST=capacity  ramps to PEAK req/s (default 100), holds, ramps down. Needs the stage
//                limit raised first; a 429 here counts as a failure.
//
// Traffic is 70% /similar, 30% /matchup, using fighter ids (names can collide -> 409).
// Writes loadtest/results/<TEST>_<RATE|PEAK>rps_<timestamp>.json, with start/end times
// that loadtest/cloudwatch_metrics.py reads to pull the server-side numbers.

import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { Rate } from "k6/metrics";

const BASE_URL = (__ENV.BASE_URL || "").replace(/\/+$/, "");
if (!BASE_URL) throw new Error("set -e BASE_URL=<api_url>");

const TEST = __ENV.TEST || "throttle";
const RATE = Number(__ENV.RATE || 20);
const PEAK = Number(__ENV.PEAK || 100);
const DURATION = __ENV.DURATION || "1m";
const HOLD = __ENV.HOLD || "2m";

const ids = new SharedArray("fighter ids", () => JSON.parse(open("./fighter_ids.json")));
const throttled = new Rate("throttled");

const scenarios = {
  throttle: {
    executor: "constant-arrival-rate",
    rate: RATE,
    timeUnit: "1s",
    duration: DURATION,
    preAllocatedVUs: 20,
    maxVUs: 100,
  },
  capacity: {
    executor: "ramping-arrival-rate",
    startRate: 5,
    timeUnit: "1s",
    preAllocatedVUs: 50,
    maxVUs: 400,
    stages: [
      { target: PEAK, duration: "1m" }, // ramp up
      { target: PEAK, duration: HOLD }, // hold at peak
      { target: 0, duration: "30s" }, // ramp down
    ],
  },
};
if (!scenarios[TEST]) throw new Error(`TEST must be one of: ${Object.keys(scenarios).join(", ")}`);

// A 429 is the right answer in the throttle test and a failure in the capacity test.
http.setResponseCallback(TEST === "throttle" ? http.expectedStatuses(200, 429) : http.expectedStatuses(200));

export const options = {
  scenarios: { [TEST]: scenarios[TEST] },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    checks: ["rate>0.99"],
    // Listing per-endpoint sub-metrics here also makes them appear in the summary.
    "http_req_duration{endpoint:similar,status:200}": ["p(95)<1000"],
    "http_req_duration{endpoint:matchup,status:200}": ["p(95)<1000"],
    throttled: TEST === "throttle" ? ["rate>0"] : ["rate==0"],
  },
};

function pick() {
  return ids[Math.floor(Math.random() * ids.length)];
}

export default function () {
  let res;
  if (Math.random() < 0.7) {
    res = http.get(`${BASE_URL}/similar?fighter=${pick()}&k=10`, { tags: { endpoint: "similar" } });
    throttled.add(res.status === 429);
    check(res, {
      "status is 200 or 429": (r) => r.status === 200 || r.status === 429,
      "similar returns 10 fighters": (r) => r.status !== 200 || r.json("similar").length === 10,
    });
  } else {
    const a = pick();
    let b = pick();
    while (b === a) b = pick();
    res = http.get(`${BASE_URL}/matchup?a=${a}&b=${b}`, { tags: { endpoint: "matchup" } });
    throttled.add(res.status === 429);
    check(res, {
      "status is 200 or 429": (r) => r.status === 200 || r.status === 429,
      "matchup returns a probability": (r) => {
        if (r.status !== 200) return true;
        const p = r.json("p_a_wins");
        return p >= 0 && p <= 1;
      },
    });
  }
}

function fmt(ms) {
  return ms === undefined ? "-" : `${ms.toFixed(1)}ms`;
}

function trendLine(label, m) {
  if (!m) return `  ${label.padEnd(26)} no data`;
  const v = m.values;
  return `  ${label.padEnd(26)} med ${fmt(v.med)}  p95 ${fmt(v["p(95)"])}  p99 ${fmt(v["p(99)"])}  max ${fmt(v.max)}`;
}

export function handleSummary(data) {
  const end = new Date();
  const start = new Date(end.getTime() - data.state.testRunDurationMs);
  const m = data.metrics;
  const reqs = m.http_reqs ? m.http_reqs.values : { count: 0, rate: 0 };
  const label = TEST === "throttle" ? `${RATE}rps` : `${PEAK}rps`;
  const stamp = start.toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const file = `loadtest/results/${TEST}_${label}_${stamp}.json`;

  const failedThresholds = Object.entries(m)
    .filter(([, metric]) => metric.thresholds && Object.values(metric.thresholds).some((t) => !t.ok))
    .map(([name]) => name);

  const text = [
    "",
    `${TEST} test, ${label}, ${start.toISOString()} -> ${end.toISOString()}`,
    `  requests                   ${reqs.count} (${reqs.rate.toFixed(1)}/s achieved)`,
    `  throttled (429)            ${((m.throttled ? m.throttled.values.rate : 0) * 100).toFixed(1)}%`,
    `  failed (unexpected status) ${((m.http_req_failed ? m.http_req_failed.values.rate : 0) * 100).toFixed(2)}%`,
    `  checks passed              ${((m.checks ? m.checks.values.rate : 0) * 100).toFixed(2)}%`,
    `  dropped iterations         ${m.dropped_iterations ? m.dropped_iterations.values.count : 0}`,
    trendLine("latency, all", m.http_req_duration),
    trendLine("latency, /similar 200", m["http_req_duration{endpoint:similar,status:200}"]),
    trendLine("latency, /matchup 200", m["http_req_duration{endpoint:matchup,status:200}"]),
    `  thresholds                 ${failedThresholds.length ? "FAILED: " + failedThresholds.join(", ") : "all passed"}`,
    `  saved                      ${file}`,
    "",
  ].join("\n");

  const record = {
    test: TEST,
    params: { rate: RATE, peak: PEAK, duration: DURATION, hold: HOLD },
    start: start.toISOString(),
    end: end.toISOString(),
    metrics: data.metrics,
  };
  return { stdout: text, [file]: JSON.stringify(record, null, 2) };
}
