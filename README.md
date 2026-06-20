# sbc
# Physics-Informed Graph Neural Networks for N-Body Systems 🌌🛰️

This repository contains a **Physics-Informed Graph Neural Network (PINN-GNN)** pipeline built using **PyTorch Geometric** designed to learn the underlying differential equations of orbital mechanics (Newtonian Gravity) directly from raw coordinate tracking data.

Instead of treating the spatial trajectories as an unstructured sequence, this approach maps the celestial layout into a directed graph structure where planets act as nodes and mutual gravitational interactions map onto message-passing channels.

---

## 🏗️ System Architecture

### 1. Data Generation (`3planet_system.py`)
Generates high-fidelity ground-truth physical systems using an analytical $N$-body differential equation solver via `scipy.integrate.solve_ivp`.
* **Configuration:** A 3-body system in a 2D Cartesian plane tracking a massive core stellar body ($m=100$) and two distinctive lightweight planetary entities ($m=1.0, 1.5$) moving at stable velocities.
* **Output:** A structural numpy matrix of size `(1000, 3, 4)` containing positions ($x, y$) and velocity vectors ($v_x, v_y$).

### 2. Deep Interaction Network Architecture
The graph pipeline abstracts the problem locally using a custom **Message Passing Neural Network (MPNN)** consisting of two isolated Multi-Layer Perceptrons (MLPs):

* **Edge MLP:** Evaluates pairwise spatial features between source and target nodes to generate abstract communication vectors. This acts as a proxy function mapping hidden gravitational force interactions:
  $$e_{ij} = \text{MLP}_{\text{edge}}(x_i \parallel x_j)$$
* **Aggregation:** Collects incoming structural message forces acting on target node $i$ via a symmetric summation operator: $\sum e_{ij}$.
* **Node MLP:** Combines aggregated force messages with the target planet's current velocity to derive the true acceleration:
  $$a_i = \text{MLP}_{\text{node}}\left(\sum_{j} e_{ij} \parallel v_i\right)$$

---

## 📊 Performance & The Autoregressive Drift Challenge

While the network successfully minimizes Mean Squared Error (MSE) down close to zero over single-step predictions ($t \rightarrow t+1$) during the training phase, evaluating the network across long-horizon evaluation cycles exposes **Rollout Drift**. 

Because micro-errors are fed sequentially back into the network as input states over an extended generation run, errors compound exponentially—resulting in the outer planet breaking conservation laws and drifting out of bounds while the unanchored central star accelerates away due to mass/inertia blindness.

---

## 🚀 Optimization Roadmap
To achieve long-term stable rollouts and extract explicit algebraic formulas ($F = G \frac{m_1 m_2}{r^2}$) via **Symbolic Regression (PySR)**, the ongoing phases implement:
1. **Latent Mass Embeddings:** Assigning a free learnable scalar variable to each unique node ID so the network implicitly deduces physical inertia.
2. **Multi-Step Recurrent Loss:** Optimizing the model over 5-10 future lookahead steps concurrently during backpropagation.
3. **Stochastic Perturbation:** Injecting low-amplitude Gaussian noise ($\sigma = 10^{-4}$) to teach the MLPs self-correcting manifold trajectories.
