#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

sudo apt update
sudo apt install -y \
  mininet \
  openvswitch-switch \
  hping3 \
  iperf3 \
  curl \
  tcpdump \
  python3 \
  python3-venv \
  python3-dev \
  build-essential \
  libffi-dev \
  libssl-dev

sudo systemctl enable --now openvswitch-switch

python3 -m venv ryu-venv
source ryu-venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-sdn.txt

echo
echo "SDN environment is ready."
echo "Run the controller with:"
echo "  source ryu-venv/bin/activate"
echo "  ./tools/run_sdn_controller.sh"
