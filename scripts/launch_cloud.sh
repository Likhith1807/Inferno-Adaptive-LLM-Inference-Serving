#!/usr/bin/env bash
# Cloud phase orchestration (CO3): provision -> sync code -> run benchmarks -> fetch results -> destroy.
#
# STATUS: not executed by the authoring session (needs your AWS credentials and quota). Review with
#   terraform -chdir=infra/aws plan
# first. The instance self-terminates after MAX_HOURS as a cost backstop even if this script dies.
#
# Usage:
#   export KEY_NAME=my-keypair KEY_FILE=~/.ssh/my-keypair.pem SSH_CIDR=$(curl -s https://checkip.amazonaws.com)/32
#   bash scripts/launch_cloud.sh up        # provision (this starts billing)
#   bash scripts/launch_cloud.sh run       # sync repo, set up env, run cloud benchmarks
#   bash scripts/launch_cloud.sh fetch     # copy results/ back
#   bash scripts/launch_cloud.sh down      # destroy everything (stops billing)
set -euo pipefail

: "${KEY_NAME:?set KEY_NAME}" "${KEY_FILE:?set KEY_FILE}" "${SSH_CIDR:?set SSH_CIDR (x.x.x.x/32)}"
TF="terraform -chdir=infra/aws"
TF_VARS=(-var "key_name=$KEY_NAME" -var "ssh_cidr=$SSH_CIDR" -var "max_hours=${MAX_HOURS:-6}"
         -var "instance_type=${INSTANCE_TYPE:-g5.12xlarge}" -var "use_spot=${USE_SPOT:-false}")
SSH_OPTS=(-i "$KEY_FILE" -o StrictHostKeyChecking=accept-new)
ip() { $TF output -raw public_ip; }

case "${1:-}" in
  up)
    $TF init -input=false
    $TF apply -input=false -auto-approve "${TF_VARS[@]}"
    echo "Instance: $(ip). It self-terminates in ${MAX_HOURS:-6} h. Run: bash scripts/launch_cloud.sh run"
    ;;
  run)
    HOST="ubuntu@$(ip)"
    until ssh "${SSH_OPTS[@]}" "$HOST" true 2>/dev/null; do sleep 5; done
    rsync -az --exclude .git --exclude results --exclude data -e "ssh ${SSH_OPTS[*]}" ./ "$HOST:~/inferno/"
    ssh "${SSH_OPTS[@]}" "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/inferno
# The DLAMI ships a recent driver, so no resolution-date pin is needed for the CUDA build.
UV_EXCLUDE_NEWER=2099-01-01T00:00:00Z bash scripts/setup_env.sh
source ~/.venvs/inferno/bin/activate
nvidia-smi -L
# Larger model + one replica per GPU. Same code path as the local study.
python -m inferno.experiments.run_sweep    --config configs/experiment_config.yaml \
    --model-config configs/model_config_cloud.yaml --reps 5
python -m inferno.experiments.run_adaptive --config configs/experiment_config.yaml \
    --model-config configs/model_config_cloud.yaml --pool-config configs/pool_config_cloud.yaml --name cloud_campaign
REMOTE
    ;;
  fetch)
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$(ip):~/inferno/results/" ./results/cloud/
    echo "Results in results/cloud/"
    ;;
  down)
    $TF destroy -input=false -auto-approve "${TF_VARS[@]}"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1;;
esac
