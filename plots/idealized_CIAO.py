import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ─── parameters ────────────────────────────────────────────────────────────────
N = 200          # number of generations on each axis
spy_color = "red"
res_color = "blue"

# ─── build idealised CIAO matrix ───────────────────────────────────────────────
# matrix[y, x]: x = spy gen (gx), y = resistance gen (gy)
ciao_ideal = np.zeros((N, N), dtype=int)
for gx in range(N):
    for gy in range(N):
        # if spy generation > resistance generation → spy “wins”
        ciao_ideal[gy, gx] = 1 if gx > gy else 0

# ─── make a 2‑color colormap ───────────────────────────────────────────────────
cmap_rb = ListedColormap([res_color, spy_color])

# ─── plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6,6))
plt.imshow(
    ciao_ideal,
    cmap=cmap_rb,
    origin="lower",
    extent=[1, N, 1, N],
    aspect="auto",
)
plt.xlabel("Spy Generation (gx)")
plt.ylabel("Resistance Generation (gy)")
plt.title("Idealized CIAO (red = spy wins, blue = res wins)")
# put ticks every 50 gens just to keep it legible
ticks = np.linspace(1, N, 5, dtype=int)
plt.xticks(ticks)
plt.yticks(ticks)
plt.grid(False)
plt.tight_layout()
plt.savefig("idealized_CIAO.png", dpi=300)
plt.show()
