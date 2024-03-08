import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

num_val = 100
max_nN = 15

# Avec heatmap quantum computing
# Étendue des valeurs pour N et n
N_range = np.linspace(1, max_nN, num_val)
n_range = np.linspace(1, max_nN, num_val)

# Grille de valeurs pour N et n
N, n = np.meshgrid(N_range, n_range)

# Calcul de la complexité pour chaque paire (N, n)
complexity = np.sqrt(2**n / N)

min_complexity_quantum = np.min(complexity)
max_complexity_quantum = np.max(complexity)

fig = go.Figure(
    data=go.Heatmap(
        z=complexity,
        x=N_range,
        y=n_range,
        colorscale="Jet",
        zsmooth="best",
    )
)
fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_heatmap.html")
