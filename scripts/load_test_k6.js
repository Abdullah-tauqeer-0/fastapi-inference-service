import http from "k6/http";
import { check } from "k6";

export const options = {
  vus: 20,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<500"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const payload = JSON.stringify({ features: [1.0, 0.5, -0.2] });
const params = {
  headers: {
    "Content-Type": "application/json",
  },
};

export default function () {
  const res = http.post(`${BASE_URL}/predict`, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}

