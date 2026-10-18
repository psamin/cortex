# Isaac Lab on Futurama (blocked)

Attempted setup: Isaac Sim 4.5.0 (pip; the only version whose wheels fit Futurama's glibc 2.34) +
IsaacLab v2.1.1 + LeIsaac v0.2.0 (SO-101 scenes), on an L40S (RTX; the H200 has no RT cores).

    sbatch -J isaac-smoke isaac_smoke.sbatch ~/cortex-isaac   # install + render two LeIsaac scenes
    sbatch -J isaac-diag  isaac_diag.sbatch  ~/cortex-isaac   # bare SimulationApp + one camera

What works: the install (15 GB venv in /dev/shm), Kit 106.5 startup (~8 s), the NVIDIA Vulkan ICD.
Fixed on the way: flatdict's build needs pkg_resources (`setuptools<70`, `--no-build-isolation`);
libGLU is missing on the node (unpacked from the Rocky 9 RPM with `rpmfile`); h5py.

Blocker: every run segfaults in `omni.kit.widget.viewport ... __enable_hydra_engine` when the RTX
renderer starts, including a bare SimulationApp with `--/rtx/verifyDriverVersion/enabled=false`.
The node runs NVIDIA driver 595.84, far newer than Isaac Sim 4.5 targets; Isaac Sim 5.x would need
glibc 2.35 (not available). Next option: NVIDIA's Isaac Sim container on a rented GPU (e.g. RunPod).
The task scenes we train on today are MuJoCo (backend/cortex/sim/tasks/).
