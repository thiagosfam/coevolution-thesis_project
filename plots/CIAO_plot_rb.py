import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load matrix
ciao_matrix = np.load("ciao_matrix_avg.npy")

# Define red = spy win, blue = resistance win
red_blue_cmap = LinearSegmentedColormap.from_list("BlueRed", ["blue", "red"])

plt.figure(figsize=(10, 8))
plt.imshow(
    ciao_matrix.T,
    cmap=red_blue_cmap,
    origin='lower',  # so lower gens are at bottom-left
    extent=[1, 200, 1, 200],  # axis ranges
    aspect='auto',
    vmin=0,
    vmax=1
)

plt.colorbar(label="Spy Win Rate")
plt.xlabel("Spy Generation (gx)")
plt.ylabel("Resistance Generation (gy)")
plt.title("CIAO: Spy[gx] vs Resistance[gy] (Red=Spy Wins, Blue=Resistance Wins)")
plt.tight_layout()
plt.show()
