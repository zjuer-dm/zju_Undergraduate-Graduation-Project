# Project Context: ETP-R1 (Vision-Language Navigation Agent)

## 1. Overview
This project implements **ETP-R1**, a graph-based VLN agent operating in continuous environments (VLN-CE). 
The core pipeline involves:
1.  **Topological Map:** Representing the environment as a graph of waypoints.
2.  **Dual-Phase Fusion Transformer (DPFT):** A custom cross-modal architecture fusing instruction text and visual graph nodes.
3.  **Action Space:** Selecting the next node (waypoint) from the graph (High-level action).
4.  **Training:** Joint Pretraining -> Online SFT (DAgger) -> Online RFT (GRPO).

---

## 2. Model Architecture Details

### A. Inputs & Encoders
* **Instruction Encoder:** * **Backbone:** RoBERTa (12 layers).
    * **Embeddings:** Word Emb + Pos Emb + Type Emb + **Task Emb** (ID: 1=R2R, 2=RxR, 3=Gemini).
    * **Output:** Text features $T = \{t_i\}_{i=1}^{N_t} \in \mathbb{R}^{N_t \times d}$.
* **Node (Visual) Encoder:**
    * **Input:** 12 panoramic views ($I^{rgb}, I^{d}$).
    * **Backbones (Frozen):** ViT-B/32 (CLIP) for RGB, ResNet-50 (PointGoal) for Depth.
    * **Fusion:** Unimodal features + View Angle Emb -> Linear -> Transformer (2 layers) -> Mean Pooling.
    * **Node Feature:** Visual Feat + Step Emb (time) + Position Emb (relative pose) + Task Emb.
    * **Output:** Graph tokens $G = \{g_i\}_{i=1}^{N_g} \in \mathbb{R}^{N_g \times d}$ (includes STOP token).

### B. Dual-Phase Fusion Transformer (DPFT)
The core reasoning module consists of two phases:

**Phase 1: Symmetric Cross-Modal Fusion**
* **Structure:** $L_s$ layers (set to 4).
* **Logic per layer:**
    1.  Bidirectional Cross-Attention (Text $\leftrightarrow$ Graph) using **unshared** weights.
    2.  Self-Attention.
    3.  Feed-Forward Network (FFN).
* **Output:** Intermediate features $T_{sym}$ and $G_{sym}$.

**Phase 2: Text-Guided Graph Refinement**
* **Purpose:** Distill text guidance into graph nodes.
* **Operation:**
    1.  **Query:** Graph features $G_{sym}$.
    2.  **Key/Value:** Text features $T_{sym}$.
    3.  $G_{guide} = \text{CrossAttn}(Q=G_{sym}, K=T_{sym}, V=T_{sym})$.
    4.  $G_{out} = \text{Concat}(G_{sym}, \text{FFN}(G_{guide}))$.
* **Final Output:** $G_{out}$ is used for action prediction.

### C. Task Heads
* **SAP Head (Single Action Prediction):** * Implementation: Simple FFN.
    * Formula: $s_i = \text{FFN}_{sap}(g_i)$ where $g_i \in G_{out}$.
    * Selects node with max score.
* **MLM Head (Auxiliary):**
    * Implementation: Simple FFN.
    * Formula: $l_j = \text{FFN}_{mlm}(t_j)$ for masked text tokens.

**Hyperparameters:** Hidden dimension $d=768$.

---

## 3. Training Paradigm & Algorithms

### A. Stage 1: Offline Joint Pretraining
* **Data:** R2R, RxR, Prevalent, RxR-Marky, **Prevalent_Gemini_Aug** (Custom generated).
* **Loss:** SAP Loss + MLM Loss (1:1 ratio).

### B. Stage 2: Online SFT (DAgger)
* **Algorithm:** Dataset Aggregation (DAgger).
* **Policy:** Mixture of Expert Action (prob $p$) and Agent Sampled Action (prob $1-p$).
* **Expert:** Global planner based on ground truth map.

### C. Stage 3: Online RFT (GRPO) - *Critical Implementation*
**Group Relative Policy Optimization (GRPO)** is used instead of PPO (Critic-free).

1.  **Sampling:** * For each instruction (prompt), sample a group of $G=8$ trajectories (answers).
    * **Dropout** must be enabled during sampling for diversity.
2.  **Reward Function ($R$):**
    * **R2R-CE:** $\mathbb{I}(d_{final}<1.5) + \text{SPL} - d_{final}/6$
    * **RxR-CE:** $\text{nDTW} + \text{SDTW} + \text{gSPL} - d_{final}/6$
    * *Note:* gSPL uses ground-truth path length, not shortest path.
3.  **Advantage Calculation:**
    * Compute rewards $r = \{r_1, ..., r_G\}$ for the group.
    * Normalize: $\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.
    * Assign this single value to all steps $t$ in trajectory $i$.
4.  **Optimization:**
    * Maximize objective with KL-divergence regularization (reference policy vs current policy).
    * Train only DPFT and SAP head.
    * Single update per batch (Iteration $\mu=1$).

---

## 4. Key Engineering Constraints
* **Graph Construction:** Sparse topological map constructed online using a waypoint predictor.
* **Navigation Stack:** High-level planner selects waypoint -> Deterministic controller executes low-level motion (Habitat Sim).
* **Differentiation:** Unlike standard LVLM methods, this model maintains full panoramic context in the graph nodes.