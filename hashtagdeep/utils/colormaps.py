""" Color palettes and colormaps for visualization.

To use the matplotlib palette, unpack the dictionary into the call to imshow:
    plt.imshow(..., **palette)

"""

import torch
from matplotlib import colors

colors_hex = [
    '#221f20',  # Class 0 – No data + Clouds and shadows
    '#cc665c',  # Class 1 – Urban
    '#d19962',  # Class 2 – Industrial / transportation
    '#dacf6a',  # Class 3 – Construction / dump cites
    '#b7d86a',  # Class 4 – Artificial non-agricultural vegetation
    '#8fd769',  # Class 5 – Annual crops
    '#8cc182',  # Class 6 – Permanent crops
    '#6fae62',  # CLass 7 – Pastures
    '#dbf5d7',  # Class 8
    '#bae0b4',  # Class 9
    '#377d22',  # Class 10 – Forests
    '#6eaea7',  # Class 11 – Herbaceous vegetation
    '#915f26',  # Class 12 – No vegetation
    '#669bd6',  # Class 13 – Wetlands
    '#2266f6',  # Class 14 – Water
]

colors_rgb = list(map(lambda c: (int(c[1:3], 16), int(c[3:5], 16), int(c[5:], 16)), colors_hex))

dfc22_labels_palette = {
    'cmap': colors.ListedColormap(colors_hex),
    'norm': colors.BoundaryNorm(boundaries=range(-1, 15), ncolors=15),
}

# this is used to visualize predictions where the prediction is a class index
# ground truth masks use -1 for class 0
dfc22_labels_color_map = torch.FloatTensor(colors_rgb[1:] + colors_rgb[:1])
