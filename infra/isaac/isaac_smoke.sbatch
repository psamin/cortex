#!/bin/bash
# Isaac Lab feasibility on Futurama: Isaac Sim 4.5.0 (pip, glibc 2.34) + IsaacLab v2.1.1 + LeIsaac v0.2.0,
# render SO-101 scenes headless and save camera videos. Needs an RTX GPU (H200 has no RT cores).
#   sbatch -J isaac-smoke isaac_smoke.sbatch <job_dir>
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
JOB_DIR="$1"
[ -d "$JOB_DIR" ] || JOB_DIR="${JOB_DIR/#\/nethome\//\/nethome-instruction\/}"
SCR="/dev/shm/isaac-${SLURM_JOB_ID}"
mkdir -p "$SCR/home" "$SCR/sys" "$JOB_DIR/out"
trap 'rm -rf "$SCR"' EXIT INT TERM
# $HOME is an ELOOP symlink on compute nodes, and Kit writes caches under it: point it at scratch.
export HOME="$SCR/home" XDG_CACHE_HOME="$SCR/cache" UV_CACHE_DIR="$SCR/uv" PIP_CACHE_DIR="$SCR/pip" \
       OMNI_KIT_ACCEPT_EULA=YES PATH="$SCR/bin:$PATH"
echo "[isaac] node=$(hostname -s) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

# The node has no libGLU (RTX's MDL compiler, neuraylib, needs it) and we can't install system
# packages: unpack the matching Rocky 9 RPM into scratch and put it on the library path.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCR/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
uv venv -q --python 3.10 --seed "$SCR/venv"
source "$SCR/venv/bin/activate"
curl -sL -o "$SCR/glu.rpm" https://dl.rockylinux.org/pub/rocky/9/AppStream/x86_64/os/Packages/m/mesa-libGLU-9.0.1-6.el9.x86_64.rpm
uv pip install -q rpmfile zstandard  # no rpm2cpio on compute nodes
python - "$SCR/glu.rpm" "$SCR/sys" <<'EOF'
import os, sys, rpmfile
with rpmfile.open(sys.argv[1]) as rpm:
    for m in rpm.getmembers():
        dest = os.path.join(sys.argv[2], m.name.lstrip("./"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = rpm.extractfile(m).read()
        if data.startswith(b"libGLU") and len(data) < 64:  # a symlink stored as its target name
            os.symlink(data.decode().strip(), dest)
        else:
            open(dest, "wb").write(data)
EOF
export LD_LIBRARY_PATH="$SCR/sys/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[isaac] libGLU: $(ls -la "$SCR/sys/usr/lib64" | tail -n +2 | awk '{print $NF, $5}' | tr '\n' ' ')"
uv pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install -q "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
echo "[isaac] isaacsim installed: $(du -sh "$SCR/venv" | cut -f1)"
# isaaclab pins flatdict==4.0.1, whose sdist needs pkg_resources at build time (gone from new setuptools)
uv pip install -q "setuptools<70" wheel
uv pip install -q --no-build-isolation flatdict==4.0.1
git clone -q -c advice.detachedHead=false --depth 1 -b v2.1.1 https://github.com/isaac-sim/IsaacLab.git "$SCR/IsaacLab"
uv pip install -q -e "$SCR/IsaacLab/source/isaaclab" -e "$SCR/IsaacLab/source/isaaclab_assets" \
  -e "$SCR/IsaacLab/source/isaaclab_tasks" imageio imageio-ffmpeg h5py --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118
git clone -q -c advice.detachedHead=false --depth 1 -b v0.2.0 https://github.com/LightwheelAI/leisaac.git "$SCR/leisaac"
uv pip install -q -e "$SCR/leisaac/source/leisaac"
A="$SCR/leisaac/assets"
mkdir -p "$A/robots" "$A/scenes"
curl -sL -o "$A/robots/so101_follower.usd" https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd
for z in v0.1.0/kitchen_with_orange.zip v0.1.2/table_with_cube.zip; do
  curl -sL -o "$SCR/scene.zip" "https://github.com/LightwheelAI/leisaac/releases/download/$z"
  python -m zipfile -e "$SCR/scene.zip" "$A/scenes/"
done
cp "$JOB_DIR/smoke_leisaac.py" "$SCR/"
cd "$SCR/leisaac"
for task in LeIsaac-SO101-LiftCube-v0 LeIsaac-SO101-PickOrange-v0; do
  rc=0
  python "$SCR/smoke_leisaac.py" --task "$task" --out "$JOB_DIR/out" --headless --enable_cameras --device cuda \
    > "$JOB_DIR/out/$task.log" 2>&1 || rc=$?
  echo "[isaac] $task exit=$rc"
  grep -E "\[isaac\]|\[Error\]|Traceback|Segmentation|Error:" "$JOB_DIR/out/$task.log" | cut -c1-300 | head -30 || true
done
echo "[isaac] DONE"
