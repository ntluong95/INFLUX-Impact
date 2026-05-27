import json
import os
from pathlib import Path

import requests

env_path = Path("zika/config/.env")
for raw in env_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == "OPENAI_API_KEY" and value.strip():
        os.environ["OPENAI_API_KEY"] = value.strip()

state_path = Path(
    "zika/data/intermediate/openai_batch/zika_headline_openai_batch_state.json"
)
state = json.loads(state_path.read_text(encoding="utf-8"))

api_key = os.environ["OPENAI_API_KEY"]
batch_id = state["batch_id"]

base_url = "https://api.openai.com/v1"

resp = requests.get(
    f"{base_url}/batches/{batch_id}",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=60,
)
resp.raise_for_status()
data = resp.json()

request_counts = data.get("request_counts") or {}
total = int(request_counts.get("total", 0) or 0)
completed = int(request_counts.get("completed", 0) or 0)
failed = int(request_counts.get("failed", 0) or 0)

print(
    json.dumps(
        {
            "batch_id": data.get("id"),
            "status": data.get("status"),
            "request_counts": request_counts,
            "remaining": total - completed - failed,
            "output_file_id": data.get("output_file_id"),
            "error_file_id": data.get("error_file_id"),
        },
        indent=2,
    )
)
