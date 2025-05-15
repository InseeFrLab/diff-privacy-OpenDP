import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.title("📊 Exploration interactive de la Loi Gaussienne entière (Integer Gaussian)")
st.markdown("""
Cette application vous permet de visualiser concrètement une **loi normale discrète** centrée sur une moyenne entière, avec un écart type spécifié.
Nous considérons ici une version discrétisée de la loi normale, souvent utilisée en **confidentialité différentielle** sous le nom de *Integer Gaussian*.
""")

# --- Entrée manuelle des paramètres ---
st.sidebar.header("⚙️ Paramètres de la loi normale entière")
mu = st.sidebar.number_input("Moyenne entière (μ)", min_value=-1_000_000, max_value=1_000_000, value=0, step=1)
sigma = st.sidebar.number_input("Écart type (σ)", min_value=0.1, max_value=10_000.0, value=3.0, step=0.1)

# Discrétisation sans doublons
raw_x_vals = np.arange(mu - 6*sigma, mu + 6*sigma + 1)
x_vals = np.round(raw_x_vals).astype(int)
unique_x, indices = np.unique(x_vals, return_inverse=True)

# Agrégation des probas sur les entiers uniques
raw_p_vals = norm.pdf(raw_x_vals, loc=mu, scale=sigma)
p_vals = np.zeros_like(unique_x, dtype=float)
for i, idx in enumerate(indices):
    p_vals[idx] += raw_p_vals[i]
p_vals /= p_vals.sum()  # Normalisation

# --- Calcul des intervalles cumulés ---
intervales = {
    "50%": 0.50,
    "75%": 0.75,
    "90%": 0.90,
    "95%": 0.95,
    "99%": 0.99
}

cdf = np.cumsum(p_vals)
cdf_dict = dict(zip(x_vals, cdf))

# --- Affichage des intervalles ---
with st.expander("📐 Intervalles de concentration autour de la moyenne"):
    for niveau, seuil in intervales.items():
        indices = np.where(cdf >= (1 - seuil) / 2)[0]
        low_idx = indices[0] if len(indices) > 0 else 0
        high_idx = np.where(cdf <= 1 - (1 - seuil) / 2)[0]
        high_idx = high_idx[-1] if len(high_idx) > 0 else len(x_vals) - 1
        st.markdown(f"- **{niveau}** des valeurs sont entre `{x_vals[low_idx]}` et `{x_vals[high_idx]}`")

# --- Affichage du graphe discret ---
fig = go.Figure()
fig.add_trace(go.Bar(x=x_vals, y=p_vals, name='Probabilités discrètes', marker_color='steelblue'))
fig.add_vline(x=mu, line=dict(color="black", dash="dash"), annotation_text=f"μ = {mu}", annotation_position="top left")

fig.update_layout(
    title="Distribution d'une loi Gaussienne entière (discrétisée)",
    xaxis_title="Valeurs entières",
    yaxis_title="Probabilité",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# --- Interprétation ---
st.markdown("""
### 🧠 Interprétation intuitive (discrétisée)

- **μ (mu)** est le centre de la distribution, ici une valeur entière.
- **σ (sigma)** mesure la dispersion : plus σ est grand, plus la masse se répartit loin de la moyenne.
- Cette version discrète est utile pour les implémentations informatiques ou les mécanismes de confidentialité différentielle (comme le *Discrete Gaussian Mechanism*).
""")