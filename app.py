import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. Configuration & Dark Neon Glassmorphism Theme
# =========================================================
st.set_page_config(
    page_title="Simulation SIR Stochastique 3D", page_icon="🦠", layout="wide"
)

# Custom CSS Inject
st.markdown(
    """
<style>
    /* Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #05020D 0%, #0D0722 50%, #030108 100%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glowing Title Banner */
    .glowing-title-box {
        background: rgba(14, 17, 23, 0.85);
        border: 2px solid #00FFFF;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.4), inset 0 0 15px rgba(112, 0, 255, 0.3);
        margin-bottom: 25px;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00FFFF, #9D4EDD, #FF0055);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sub-title {
        color: #A7F3D0;
        font-size: 1.1rem;
        margin-top: 8px;
        letter-spacing: 1px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: #0E1117 !important;
        border: 2px solid rgba(0, 255, 255, 0.4) !important;
        border-radius: 14px !important;
        padding: 15px !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.15) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00FFFF !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.6) !important;
        font-weight: 800 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title Header
st.markdown(
    """
<div class="glowing-title-box">
    <h1 class="main-title">🦠 Simulation Stochastique du Modèle SIR</h1>
    <div class="sub-title">Analyse 3D Interactive & Simulations Monte-Carlo (Python & Plotly)</div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2. Sidebar Controls
# =========================================================
st.sidebar.header("⚙️ Paramètres de Simulation")
N = st.sidebar.slider("Nombre d'individus (N)", 10, 50, 15)
p_infect = st.sidebar.slider("Probabilité d'infection (p)", 0.0, 1.0, 0.3, 0.05)
t_max = st.sidebar.slider("Durée (Itérations)", 3, 20, 5)
num_sims = st.sidebar.slider("Simulations Monte-Carlo", 5, 100, 20)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **Astuce :** La structure 3D contient désormais un **réseau filaire"
    " volumétrique (Grille 3D & Lignes fantômes)** pour visualiser la"
    " profondeur spatiale."
)

# =========================================================
# 3. Stochastic SIR Simulation Core
# =========================================================
G = nx.erdos_renyi_graph(N, p_infect, seed=42)
pos_3d = nx.spring_layout(G, dim=3, seed=42)


def simulate_sir(N_nodes, steps, p, G_graph):
  states = [0] * N_nodes  # 0=Susceptible, 1=Infecté, 2=Rétabli
  states[0] = 1  # Patient zéro initial (Infecté)

  history_S, history_I, history_R = [], [], []
  for t in range(steps + 1):
    history_S.append(states.count(0))
    history_I.append(states.count(1))
    history_R.append(states.count(2))

    if t == steps:
      break

    new_states = states.copy()
    for node in G_graph.nodes():
      if states[node] == 0:  # Susceptible
        infected_neighbors = sum(
            1 for neighbor in G_graph.neighbors(node) if states[neighbor] == 1
        )
        if infected_neighbors > 0 and random.random() < p:
          new_states[node] = 1
      elif states[node] == 1:  # Infecté
        if random.random() < 0.5:  # Guérison 50%
          new_states[node] = 2
    states = new_states
  return history_S, history_I, history_R, states


# Single Run
S, I, R, final_states = simulate_sir(N, t_max, p_infect, G)
t_axis = list(range(len(S)))

# =========================================================
# 4. Section 1 : 3D Network Graph with Volumetric Wireframe Grid
# =========================================================
st.markdown(
    "### 🌐 1. Réseau Spatial 3D (Structure Filaire & Grille Volumétrique"
    " Fantôme)"
)

# 4.1 --- Génération de la Grille 3D Volumétrique ---
min_x = min(pos_3d[n][0] for n in G.nodes()) - 0.2
max_x = max(pos_3d[n][0] for n in G.nodes()) + 0.2
min_y = min(pos_3d[n][1] for n in G.nodes()) - 0.2
max_y = max(pos_3d[n][1] for n in G.nodes()) + 0.2
min_z = min(pos_3d[n][2] for n in G.nodes()) - 0.2
max_z = max(pos_3d[n][2] for n in G.nodes()) + 0.2

grid_x, grid_y, grid_z = [], [], []

# Cube englobant 3D
for x in [min_x, max_x]:
  for y in [min_y, max_y]:
    grid_x.extend([x, x, None])
    grid_y.extend([y, y, None])
    grid_z.extend([min_z, max_z, None])

for x in [min_x, max_x]:
  for z in [min_z, max_z]:
    grid_x.extend([x, x, None])
    grid_y.extend([min_y, max_y, None])
    grid_z.extend([z, z, None])

for y in [min_y, max_y]:
  for z in [min_z, max_z]:
    grid_x.extend([min_x, max_x, None])
    grid_y.extend([y, y, None])
    grid_z.extend([z, z, None])

# Axes centraux virtuels
mid_x, mid_y, mid_z = (
    (min_x + max_x) / 2,
    (min_y + max_y) / 2,
    (min_z + max_z) / 2,
)
grid_x.extend([min_x, max_x, None, mid_x, mid_x, None, mid_x, mid_x, None])
grid_y.extend([mid_y, mid_y, None, min_y, max_y, None, mid_y, mid_y, None])
grid_z.extend([mid_z, mid_z, None, mid_z, mid_z, None, min_z, max_z, None])

wireframe_grid_trace = go.Scatter3d(
    x=grid_x,
    y=grid_y,
    z=grid_z,
    line=dict(color="rgba(0, 255, 255, 0.25)", width=1.5, dash="dash"),
    hoverinfo="none",
    mode="lines",
    name="Grille Spatiale 3D",
)

# 4.2 --- Connexions entre les nœuds du réseau ---
edge_x, edge_y, edge_z = [], [], []
for edge in G.edges():
  x0, y0, z0 = pos_3d[edge[0]]
  x1, y1, z1 = pos_3d[edge[1]]
  edge_x.extend([x0, x1, None])
  edge_y.extend([y0, y1, None])
  edge_z.extend([z0, z1, None])

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    line=dict(color="rgba(100, 200, 255, 0.75)", width=2.5),
    hoverinfo="none",
    mode="lines",
    name="Liaisons du Réseau",
)

# 4.3 --- Nœuds du réseau ---
node_x = [pos_3d[node][0] for node in G.nodes()]
node_y = [pos_3d[node][1] for node in G.nodes()]
node_z = [pos_3d[node][2] for node in G.nodes()]

colors = [
    "#FF0000"
    if node == 0
    else ("#00FF00" if final_states[node] == 2 else "#00FFFF")
    for node in G.nodes()
]
sizes = [15 if node == 0 else 10 for node in G.nodes()]
labels = [
    f"Nœud {node} :"
    f" {'Patient Zéro (Infecté)' if node == 0 else ('Guéri' if final_states[node]==2 else 'Sain')}"
    for node in G.nodes()
]

node_trace = go.Scatter3d(
    x=node_x,
    y=node_y,
    z=node_z,
    mode="markers+text",
    hoverinfo="text",
    text=[str(n) for n in G.nodes()],
    textposition="top center",
    hovertext=labels,
    marker=dict(
        size=sizes,
        color=colors,
        opacity=0.95,
        line=dict(color="#FFFFFF", width=1.5),
    ),
)

fig_3d = go.Figure(data=[wireframe_grid_trace, edge_trace, node_trace])
fig_3d.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 255, 255, 0.15)",
            showbackground=True,
            backgroundcolor="rgba(14, 17, 23, 0.5)",
            zerolinecolor="rgba(0, 255, 255, 0.3)",
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 255, 255, 0.15)",
            showbackground=True,
            backgroundcolor="rgba(14, 17, 23, 0.5)",
            zerolinecolor="rgba(0, 255, 255, 0.3)",
            showticklabels=False,
            title="",
        ),
        zaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 255, 255, 0.15)",
            showbackground=True,
            backgroundcolor="rgba(14, 17, 23, 0.5)",
            zerolinecolor="rgba(0, 255, 255, 0.3)",
            showticklabels=False,
            title="",
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, b=0, t=0),
    height=600,
    showlegend=False,
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# =========================================================
# 5. Section 2 : Curves & Monte-Carlo Analytics
# =========================================================
col1, col2 = st.columns(2)

with col1:
  st.markdown("### 📈 2. Évolution Temporelle SIR")
  fig_sir = go.Figure()
  fig_sir.add_trace(
      go.Scatter(
          x=t_axis,
          y=S,
          mode="lines+markers",
          name="Susceptibles (S)",
          line=dict(color="#00FFFF", width=4),
          marker=dict(size=7),
      )
  )
  fig_sir.add_trace(
      go.Scatter(
          x=t_axis,
          y=I,
          mode="lines+markers",
          name="Infectés (I)",
          line=dict(color="#FF0000", width=4),
          marker=dict(size=7),
      )
  )
  fig_sir.add_trace(
      go.Scatter(
          x=t_axis,
          y=R,
          mode="lines+markers",
          name="Rétablis (R)",
          line=dict(color="#00FF00", width=4),
          marker=dict(size=7),
      )
  )

  fig_sir.update_layout(
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(14, 17, 23, 0.7)",
      xaxis=dict(
          title="Itérations (t)",
          color="#94A3B8",
          gridcolor="rgba(255,255,255,0.1)",
      ),
      yaxis=dict(
          title="Nombre d'individus",
          color="#94A3B8",
          gridcolor="rgba(255,255,255,0.1)",
      ),
      legend=dict(font=dict(color="#E2E8F0")),
      height=450,
      margin=dict(l=20, r=20, b=20, t=20),
  )
  st.plotly_chart(fig_sir, use_container_width=True)

with col2:
  st.markdown(f"### 📊 3. Enveloppe Monte-Carlo ({num_sims} Simulations)")
  all_I_pics = []
  all_trajectories = []
  fig_mc = go.Figure()

  for _ in range(num_sims):
    _, hist_I, _, _ = simulate_sir(N, t_max, p_infect, G)
    all_trajectories.append(hist_I)
    all_I_pics.append(max(hist_I))
    fig_mc.add_trace(
        go.Scatter(
            x=t_axis,
            y=hist_I,
            mode="lines",
            line=dict(color="rgba(157, 78, 221, 0.3)", width=2),
            showlegend=False,
        )
    )

  mean_trajectory = np.mean(all_trajectories, axis=0)
  moyenne_pic = np.mean(all_I_pics)
  variance_pic = np.var(all_I_pics)

  fig_mc.add_trace(
      go.Scatter(
          x=t_axis,
          y=mean_trajectory,
          mode="lines+markers",
          name="Moyenne des infectés",
          line=dict(color="#FF0000", width=4),
          marker=dict(size=8),
      )
  )

  fig_mc.update_layout(
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(14, 17, 23, 0.7)",
      xaxis=dict(
          title="Itérations (t)",
          color="#94A3B8",
          gridcolor="rgba(255,255,255,0.1)",
      ),
      yaxis=dict(
          title="Infectés (I)", color="#94A3B8", gridcolor="rgba(255,255,255,0.1)"
      ),
      legend=dict(font=dict(color="#E2E8F0")),
      height=450,
      margin=dict(l=20, r=20, b=20, t=20),
  )
  st.plotly_chart(fig_mc, use_container_width=True)

# =========================================================
# 6. Section 3 : Metrics & Neon Conclusion Box
# =========================================================
col_m1, col_m2, col_m3 = st.columns([1, 1, 2])

with col_m1:
  st.metric(label="Moyenne au pic", value=f"{moyenne_pic:.2f}")

with col_m2:
  st.metric(label="Variance au pic", value=f"{variance_pic:.2f}")

with col_m3:
  st.markdown(
      """
    <div style="
        background-color: #0E1117;
        border: 2px solid #00FFFF;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 0 15px #00FFFF;
    ">
        <h4 style="color: #00FFFF; margin-top: 0; font-weight: 800; font-size: 1.1rem; text-shadow: 0 0 8px #00FFFF;">
            💡 Conclusion Scientifique
        </h4>
        <p style="color: #00FFFF; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0;">
            La <strong>stochasticité (le hasard)</strong> génère une enveloppe de trajectoires réelles. L'analyse Monte-Carlo montre l'écart entre la trajectoire moyenne et les scénarios extrêmes. Chaque simulation produit une courbe unique, prouvant la nécessité d'une approche stochastique.
        </p>
    </div>
    """,
      unsafe_allow_html=True,
  )