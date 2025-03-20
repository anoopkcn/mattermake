from pymatgen.analysis.local_env import EconNN
import torch
import numpy as np

from pymatgen.core import Structure
from pathlib import Path

# import polars as pl
# import matplotlib.pyplot as plt

# df = pl.read_csv("~/Downloads/alex_mp_20/alex_mp_20/train.csv", null_values=["nan"])
# print(df.columns)
# print(df.shape)

# df_selected = df.select(["material_id", "cif", "dft_band_gap"])
# # Drop rows with nan or null values

# df2 = df.drop_nulls()
# print(df2.shape)

# df_large_gaps = df.filter(pl.col("dft_band_gap") > 0.5)
# print(df_large_gaps.shape[0])

# plt.figure(figsize=(10, 6))
# plt.hist(df2["dft_band_gap"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
# plt.xlabel("Band Gap (eV)")
# plt.ylabel("Count")
# plt.title("Distribution of DFT Band Gaps")
# plt.grid(True, alpha=0.3)
# plt.show()

# Path to the CIF file
file_path = Path("~/Downloads/MAPI.cif").expanduser()
structure = Structure.from_file(file_path)
print(f"Structure summary: {structure}")
print(f"Number of sites: {len(structure)}")
print(f"Lattice parameters: {structure.lattice.parameters}")

sites = structure.sites
atomic_numbers = [site.specie.Z for site in sites]
atomic_numbers

num_sites = len(sites)

node_features = torch.zeros((num_sites, 100))
for i, z in enumerate(atomic_numbers):
    node_features[i, z - 1] = 1

frac_coords = np.array([site.frac_coords for site in sites])

node_features = np.hstack([node_features, frac_coords])
nn_finder = EconNN()
neighbors = nn_finder.get_nn_info(structure, 0)
neighbors
sites[24]


neighbors = nn_finder.get_nn_info(structure, 1)
edge_features = []
edge_index = []
for neighbor in neighbors:
    j = neighbor["site_index"]
    image = neighbor.get("image", [0, 0, 0])
    distance = neighbor["weight"]
    edge_index.append([24, j])
    edge_features.append([distance] + list(image))
print(edge_index[0], edge_features[0])

neighbors[0]
structure.lattice.parameters
