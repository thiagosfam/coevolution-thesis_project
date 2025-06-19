import numpy as np
import matplotlib.pyplot as plt

# Load your matrix
ciao_matrix = np.load("ciao_matrix_avg.npy")

plt.figure(figsize=(10, 8))
plt.imshow(
    ciao_matrix, 
    cmap='Greys_r',     # Reversed grayscale: white = high (spy wins), black = low (resistance wins)
    origin='lower', 
    extent=[1, 200, 1, 200],
    aspect='auto',
    vmin=0,
    vmax=1
)
plt.colorbar(label="Spy Win Rate (White = 100%)")
plt.title("CIAO Matrix (Black = Resistance Wins, White = Spies Win)")
plt.xlabel("Spy Generation (gx)")
plt.ylabel("Resistance Generation (gy)")
plt.tight_layout()
plt.show()

gx, gy = np.indices(ciao_matrix.shape)
spy_triangle = ciao_matrix[gx > gy].mean()
res_triangle = ciao_matrix[gx < gy].mean()

print(f"Spy Triangle Mean (gx > gy): {spy_triangle:.3f}")
print(f"Resistance Triangle Mean (gx < gy): {res_triangle:.3f}")