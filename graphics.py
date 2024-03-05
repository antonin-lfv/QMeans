import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

num_val = 100
max_nN = 15

# Avec heatmap quantum computing
# Étendue des valeurs pour N et n
N_range = np.linspace(1, max_nN, num_val)  # N de 1 à 1024
n_range = np.linspace(1, max_nN, num_val)  # n de 1 à 20

# Grille de valeurs pour N et n
N, n = np.meshgrid(N_range, n_range)

# Calcul de la complexité pour chaque paire (N, n)
complexity = np.maximum(
    (np.pi / 4) * np.sqrt(2**n / N), (np.pi / 4) * np.sqrt(N) - 0.5
)

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

# Avec contour plot quantum computing
fig = go.Figure(
    data=go.Contour(
        z=complexity,
        x=N_range,
        y=n_range,
        colorscale="Jet",
        contours=dict(
            start=min_complexity_quantum,
            end=max_complexity_quantum,
            size=0.5,
            showlabels=True,
            labelfont=dict(  # label font properties
                size=10,
                color="white",
            ),
        ),
    )
)
fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_contour.html")


# Avec heatmap classical computing

# Étendue des valeurs pour N et n, comme dans l'exemple précédent
N_range_classic = np.linspace(1, max_nN, num_val)  # N de 1 à 10
n_range_classic = np.linspace(1, max_nN, num_val)  # n de 1 à 10

# Grille de valeurs pour N et n
N_classic, n_classic = np.meshgrid(N_range_classic, n_range_classic)

# Calcul de la complexité pour chaque paire (N, n) pour une recherche classique
# Ici, la complexité ne dépend que de N, donc nous ignorons n dans le calcul
complexity_classic = N_classic  # La complexité est simplement O(N)

# gradient bar from min_complexity_quantum to max_complexity_quantum

fig = go.Figure(
    data=go.Heatmap(
        z=complexity_classic,
        x=N_range_classic,
        y=n_range_classic,
        colorscale="Jet",
        zsmooth="best",
    )
)
fig.data[0].update(zmin=min_complexity_quantum, zmax=max_complexity_quantum)
fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_heatmap_classic.html")

# Avec contour plot classical computing
fig = go.Figure(
    data=go.Contour(
        z=complexity_classic,
        x=N_range_classic,
        y=n_range_classic,
        colorscale="Jet",
        contours=dict(
            start=min_complexity_quantum,
            end=max_complexity_quantum,
            size=0.5,
            showlabels=True,
            labelfont=dict(  # label font properties
                size=10,
                color="white",
            ),
        ),
    )
)
fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_contour_classic.html")


# Now we compute the difference between the two complexities
# We will use the same N and n values as before
difference = complexity_classic - complexity
# scale between -1 and 1
difference = difference
# plot the difference (if negative classical is better, if positive quantum is better)
# plot the heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=difference,
        x=N_range,
        y=n_range,
        zsmooth="best",
        colorscale="RdBu",
    )
)

fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_difference.html")


# Transform the difference into a binary value
# 0 if classical is better, 1 if quantum is better
binary_difference = np.zeros_like(difference)
binary_difference[difference > 0] = 1

# only two colors, red and blue
colors = ["red", "blue"]
# plot the heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=binary_difference,
        x=N_range,
        y=n_range,
        colorscale=colors,
        zsmooth="best",
    )
)

fig.update_layout(
    xaxis_title="N",
    yaxis_title="n",
    # size of the axis title
    font=dict(size=25, color="black", family="Noto Sans"),
)

plot(fig, filename="complexity_binary_difference.html")
