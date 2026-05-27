#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_paperviz.sh — Wrapper to run PaperVizAgent for each diagram spec.
#
# Usage:
#   ./run_paperviz.sh [MODE] [MAX_ROUNDS]
#
# Defaults:
#   MODE=dev_full   MAX_ROUNDS=3
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERVIZ_DIR="$(cd "${SCRIPT_DIR}/../../papervizagent" && pwd)"
DATA_DIR="${PAPERVIZ_DIR}/data/PaperBananaBench/diagram"
RESULT_DIR="${PAPERVIZ_DIR}/results/PaperBananaBench_diagram"
MODE="${1:-dev_full}"
MAX_ROUNDS="${2:-3}"

# ── activate PaperVizAgent venv ──────────────────────────────────────────
source "${PAPERVIZ_DIR}/.venv/bin/activate"

if ! python -c "import aiofiles, PIL, numpy" >/dev/null 2>&1; then
  echo "PaperVizAgent dependencies are missing. Run:" >&2
  echo "  cd ${PAPERVIZ_DIR} && source .venv/bin/activate && uv pip install -r requirements.txt" >&2
  exit 1
fi

# ── load .env ────────────────────────────────────────────────────────────
if [[ -f "${PAPERVIZ_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PAPERVIZ_DIR}/.env"
  set +a
fi

: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY in ${PAPERVIZ_DIR}/.env or the shell.}"

if [[ ! -f "${PAPERVIZ_DIR}/configs/model_config.yaml" ]]; then
  echo "Missing ${PAPERVIZ_DIR}/configs/model_config.yaml." >&2
  echo "Create it from configs/model_config.template.yaml and set model names." >&2
  exit 1
fi

# ── prepare directories ─────────────────────────────────────────────────
mkdir -p "${DATA_DIR}/images" "${RESULT_DIR}" "${SCRIPT_DIR}/outputs"

# ── generate input JSON specs ────────────────────────────────────────────
python "${SCRIPT_DIR}/prepare_inputs.py"

# ── stage reference images into PaperVizAgent data dir ───────────────────
export REF_DIR="${SCRIPT_DIR}/references" DATA_DIR
python - <<'PY'
import json, os, shutil
from pathlib import Path

ref_dir = Path(os.environ["REF_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
images_dir = data_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

entries = []
for path in sorted(ref_dir.glob("*")):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    target = images_dir / path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(path.resolve())
    except OSError:
        shutil.copy2(path, target)
    title = path.stem.replace("_", " ").replace("-", " ")
    entries.append({
        "id": path.stem,
        "content": f"Reference scientific methodology diagram: {title}.",
        "visual_intent": title,
        "path_to_gt_image": f"images/{path.name}",
    })

(data_dir / "ref.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
(data_dir / "agent_selected_12.json").write_text(
    json.dumps(entries[:12], indent=2), encoding="utf-8"
)
PY

# ── loop over input specs ────────────────────────────────────────────────
shopt -s nullglob
input_files=( "${SCRIPT_DIR}"/inputs/*.json )
if (( ${#input_files[@]} == 0 )); then
  echo "No JSON specs found under ${SCRIPT_DIR}/inputs." >&2
  exit 1
fi

for input_file in "${input_files[@]}"; do
  name="$(basename "${input_file}" .json)"
  split_file="${DATA_DIR}/${name}.json"

  echo "──────────────────────────────────────────────────────────"
  echo "  Generating diagram: ${name}"
  echo "──────────────────────────────────────────────────────────"

  # Convert our spec format into PaperVizAgent split format
  export INPUT_FILE="${input_file}" SPLIT_FILE="${split_file}" MAX_ROUNDS
  python - <<'PY'
import json, os
from pathlib import Path

spec = json.loads(Path(os.environ["INPUT_FILE"]).read_text(encoding="utf-8"))
style = f"{spec['figure_type']} {spec['style_notes']}".lower()
aspect = "4:5" if "vertical" in style else "16:9" if "horizontal" in style else "1:1"

payload = [{
    "filename": Path(os.environ["INPUT_FILE"]).stem,
    "caption": spec["figure_caption"],
    "content": "\n".join([
        f"Figure type: {spec['figure_type']}",
        spec["method_content"],
        f"Style notes: {spec['style_notes']}",
    ]),
    "visual_intent": spec["figure_caption"],
    "additional_info": {"rounded_ratio": aspect},
    "max_critic_rounds": int(os.environ["MAX_ROUNDS"]),
}]

Path(os.environ["SPLIT_FILE"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  # Run PaperVizAgent
  python "${PAPERVIZ_DIR}/main.py" \
    --dataset_name "PaperBananaBench" \
    --task_name "diagram" \
    --split_name "${name}" \
    --exp_mode "${MODE}" \
    --retrieval_setting "manual" \
    --max_critic_rounds "${MAX_ROUNDS}"

  # Decode result image to PNG
  export RESULT_DIR INPUT_NAME="${name}" MODE OUTPUT_FILE="${SCRIPT_DIR}/outputs/${name}.png"
  python - <<'PY'
import base64, json, os
from io import BytesIO
from pathlib import Path
from PIL import Image

result_dir = Path(os.environ["RESULT_DIR"])
name = os.environ["INPUT_NAME"]
mode = os.environ["MODE"]
matches = sorted(result_dir.glob(f"*_manualret_{mode}_{name}.json"))
if not matches:
    raise SystemExit(f"No result JSON found for {name}")

items = json.loads(matches[-1].read_text(encoding="utf-8"))
if not items:
    raise SystemExit(f"Empty result JSON for {name}")

item = items[0]
key = item.get("eval_image_field")
fallbacks = [
    "target_diagram_critic_desc2_base64_jpg",
    "target_diagram_critic_desc1_base64_jpg",
    "target_diagram_critic_desc0_base64_jpg",
    "target_diagram_stylist_desc0_base64_jpg",
    "target_diagram_desc0_base64_jpg",
    "vanilla_diagram_base64_jpg",
]
if not key or key not in item:
    key = next((c for c in fallbacks if item.get(c)), None)
if not key:
    raise SystemExit(f"No rendered image found for {name}")

image = Image.open(BytesIO(base64.b64decode(item[key]))).convert("RGB")
image.save(Path(os.environ["OUTPUT_FILE"]), format="PNG")
print(f"  saved {os.environ['OUTPUT_FILE']}")
PY

done

echo ""
echo "All diagrams generated in ${SCRIPT_DIR}/outputs/"
