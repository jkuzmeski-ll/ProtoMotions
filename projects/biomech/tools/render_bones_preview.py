import sys
from pathlib import Path

import numpy as np
import mujoco

XML = "protomotions/data/assets/mjcf/biomech_rajagopal.xml"
OUT = Path("projects/biomech/docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

m = mujoco.MjModel.from_xml_path(XML)
m.vis.global_.offwidth = 1300
m.vis.global_.offheight = 1300
d = mujoco.MjData(m)

# Model is in OpenSim Y-up frame; rotate the free root +90 deg about X so +Y (head)
# maps to world +Z, i.e. the skeleton stands upright in MuJoCo's Z-up render world.
d.qpos[:3] = [0.0, 0.0, 0.95]
d.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # wxyz, +90 about X
mujoco.mj_forward(m, d)

W, H = 900, 1300
views = {"front": 90.0, "side": 0.0, "oblique": 45.0}
try:
    r = mujoco.Renderer(m, height=H, width=W)
except Exception as exc:  # noqa: BLE001
    print("RENDER_CONTEXT_FAILED:", exc)
    sys.exit(3)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.9]
cam.distance = 3.0
cam.elevation = -8.0

try:
    from PIL import Image
    save = lambda arr, p: Image.fromarray(arr).save(p)  # noqa: E731
except Exception:  # noqa: BLE001
    import imageio.v2 as imageio
    save = lambda arr, p: imageio.imwrite(p, arr)  # noqa: E731

for name, az in views.items():
    cam.azimuth = az
    r.update_scene(d, cam)
    px = r.render()
    out = OUT / f"bones_preview_{name}.png"
    save(px, str(out))
    print("wrote", out)
