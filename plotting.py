from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# For coloring attention matrices
selected_colors = plt.cm.tab20b.colors[1::2]
ATTENTION_SCORE_CMAP = ListedColormap(selected_colors)