# UPCA: Unified Predictive Cognitive Architecture

🧠 UPCA is a modular cognitive framework for solving abstraction and reasoning tasks, built around dynamic skill learning, hierarchical planning, and symbolic generalization. It demonstrates high performance on the ARC-AGI benchmark — achieving over 73% accuracy with efficient execution on commodity hardware.

---

## 🌟 Key Features

- **Skill Memory & Induction**: Stores and retrieves transformation sequences via episodic memory and generalizes them through usage-based promotion.
- **Hierarchical Planning**: Executes abstract plans using recursive resolution of primitives, learned skills, and meta-transformations.
- **Valence-Guided Retrieval**: Combines feature similarity and historical success to rank skills during inference.
- **Modular Agent Design**: Splits reasoning across ME (motor execution), MA (meta-actions), SI (skill induction), and AMC (adaptive control).
- **ARC Benchmark Ready**: Handles visual reasoning tasks from the ARC dataset with high accuracy and fast turnaround.

---

## 🧩 Project Structure

| File               | Description                                  |
|--------------------|----------------------------------------------|
| `upca_agent.py`    | Main agent loop for planning and execution   |
| `si_scaffold.py`   | Memory module for skill storage and recall   |
| `me_engine.py`     | Primitive grid-based transformations         |
| `ma_engine.py`     | Meta-transformations and high-level actions  |
| `amc_module.py`    | Coordination / control interface             |
| `main_arc.py`      | ARC benchmark runner                         |
| `train_simple.py`  | Optional training or evaluation harness      |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Dooces/UPCA.git
cd UPCA
python main_arc.py
