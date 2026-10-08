#!/usr/bin/env bash
# LOCAL (ubr-g1): 4-GPU pipeline smoke of the G1 hug Edge recipe.
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_g1_hug_edge_smoke.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
export G1_HUG_ROOT="${G1_HUG_ROOT:-}" G1_HUG_STATS="${G1_HUG_STATS:-$G1_HUG_ROOT/action_stats.json}"
EXTRA_DATASET_CHECK='[[ -f "$G1_HUG_ROOT/meta/info.json" && -f "$G1_HUG_STATS" ]] || { echo "ERROR: G1_HUG_ROOT needs meta/info.json and G1_HUG_STATS a stats JSON" >&2; exit 1; }'
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
