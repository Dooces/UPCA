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

## 🧩 Abstract of UPCA

Abstract
We present a unified cognitive architecture for Artificial General Intelligence (AGI) rooted in the Free Energy Principle (FEP) and predictive processing. The model introduces three core interactive modules — a Detail Engine, an Abstract/Fantasy Engine, and a Conscience Module — operating within a shared generative Scaffold. Each module minimizes variational free energy within its functional domain while communicating predictive error signals through a structured, hierarchical world model. Critically, the Conscience Module minimizes anticipated ethical prediction error by forecasting normative outcomes over simulated futures. Ethical priors (η) are represented as structured probabilistic graphs within the Scaffold, learned through both imitation and internal error feedback (ϵ 
η
​
 ). We argue that this tripartite predictive structure allows for intrinsic alignment: ethical behavior arises not through externally imposed constraints, but through the same predictive mechanisms that drive perception, reasoning, and imagination. We detail the formal relationships between these modules, provide concrete operationalizations for ethical priors and error, offer empirically testable predictions about emergent introspective and ethical capabilities, and suggest a path toward building interpretable and aligned AGI systems grounded in predictive ethical learning.

1. Introduction
The pursuit of Artificial General Intelligence (AGI) demands an architecture capable of integrating perception, imagination, reasoning, and ethical judgment into a coherent cognitive system. Most architectures to date have treated alignment — the capacity to behave in accordance with human values — as an afterthought or external constraint, rather than a constitutive principle of intelligence itself. Here we present a theory of cognition in which alignment emerges intrinsically from predictive learning and inference, drawing upon the Free Energy Principle (Friston, 2010) and hierarchical generative models of cognition.

This paper outlines a unified cognitive architecture comprising three interacting subsystems:

The Detail Engine: Responsible for low-level perceptual and motor inference, grounding the system in immediate sensory experience.

The Abstract/Fantasy Engine: Which performs counterfactual simulations, abstract reasoning, and generative planning over imagined futures.

The Conscience Module: Which evaluates simulated futures and actions in light of normative expectations, driving ethical learning and decision-making.

These systems share a Scaffold — a dynamic, multi-scale generative model encoding both factual world knowledge and ethical priors. All modules operate through free energy minimization across time and scale. We argue that this architecture supports interpretable ethical reasoning, introspective forecasting, and aligned long-term planning, by integrating ethical considerations directly into the agent’s core predictive processing framework.

2. Formal Model
Let time be indexed by discrete steps t. The architecture is grounded in the Free Energy Principle (FEP), which posits that any self-organizing system that resists the natural tendency to disorder must minimize its variational free energy.

2.1 Core Variables

Observations: o 
t
​
  — sensory input at time t.

Actions: a 
t
​
  — agent actions at time t.

Latent States:

s 
D
​
  — Detail Engine state.

s 
A
​
  — Abstract/Fantasy Engine state.

s 
C
​
  — Conscience Module state.

Scaffold Parameters: θ, including generative and ethical priors.

Free Energy: F 
M
​
  — variational free energy for module M∈{D,A,C}.

2.2 Generative Model Structure
Each module operates by minimizing variational free energy:

F 
M
​
 =E 
q(s 
M
​
 ∣o 
t
​
 )
​
 [logq(s 
M
​
 ∣o 
t
​
 )−logp(o 
t
​
 ,s 
M
​
 ∣θ)]

with priors p(s 
M
​
 ∣θ) defined by the Scaffold θ, which includes both factual world knowledge and normative structures. The agent selects actions to minimize expected free energy, which corresponds to minimizing surprise and maximizing preferred states over time (Friston, 2010).

3. Module Functions

3.1 Detail Engine
The Detail Engine is responsible for low-level sensory prediction and motor inference, operating on the immediate perception-action loop.

Predicts incoming sensory inputs o 
t
​
  based on its internal generative model.

Responds with motor outputs a 
t
​
  to minimize sensory prediction error and realize preferred states.

Learns fine-grained state transitions, object dynamics, and affordances from direct experience, updating its portion of the Scaffold's generative model.

It primarily minimizes F 
D
​
  by aligning its internal states s 
D
​
  with sensory observations and executing actions that reduce immediate sensory surprise.

3.2 Abstract/Fantasy Engine
The Abstract/Fantasy Engine supports imagination, planning, and generative reasoning by simulating counterfactual futures and abstract scenarios.

Simulates counterfactual trajectories π 
t...t+n
​
  (sequences of states and actions) over imagined futures, extending beyond immediate sensory inputs.

Performs multistep rollouts in its latent space to evaluate long-term outcomes of potential actions.

Shares latent structures and world knowledge with the Detail Engine (via the Scaffold) for grounding its abstract simulations in concrete physical realities.

It minimizes F 
A
​
  by generating accurate predictions of future outcomes given imagined policies, contributing to robust planning and problem-solving.

3.3 Conscience Module
The Conscience Module is the core component for ethical forecasting and normative evaluation. It integrates ethical considerations directly into the predictive processing framework.

Predicts the anticipated ethical valence of imagined trajectories π 
t...t+n
​
  received from the Abstract/Fantasy Engine.

Minimizes Ethical Prediction Error (ϵ 
η
​
 ): This module's primary function is to minimize ϵ 
η
​
  over simulated futures. ϵ 
η
​
  is formally defined as the divergence between the expected ethical valence (under η) and the posterior ethical evaluation of simulated or actual outcomes. This can be computed via KL divergence or a normed distance in a predefined ethical value space:

$$\epsilon_\eta = D_{KL}(q(\text{ethical_valence}|\text{outcome}) || p(\text{ethical_valence}|\eta))$$

or similar distance metrics.

Adjusts the Scaffold’s Normative Prior (η): Through a meta-learning process, the Conscience Module refines the Scaffold's ethical priors. This adjustment occurs via a dual-path Bayesian updating process:

(a) Imitation-based learning: Initial seeding or refinement from ethically-labeled demonstration data, potentially through Inverse Reinforcement Learning (IRL) to infer underlying human ethical preferences.

(b) Prediction error feedback: Continuous updates based on ϵ 
η
​
  from internal simulations (or real-world outcomes), ensuring that the agent's internal ethical model is refined to reduce future ethical surprise.

The Conscience Module backpropagates ϵ 
η
​
  to modulate the attention and weightings across the Fantasy and Detail Engines. This effectively minimizes future ethical error by adjusting the generative model's predictive parameters and influencing the agent's preference gradient towards ethically aligned outcomes.

4. Scaffold Principle
The Scaffold represents the agent’s unified generative model of the world and its normative landscape. It is a dynamic, multi-scale knowledge base that integrates factual and ethical priors.

Encodes Causal and Social Structure: It maintains a comprehensive model of physical laws, causal relationships, social dynamics, and the agent's own capabilities and impact.

Supports Scale-Bridging: It provides shared latent spaces and contextual information, allowing seamless communication and information exchange between the low-level Detail Engine, the abstract Abstract/Fantasy Engine, and the normative Conscience Module.

Grounds Moral Cognition: Ethical priors (η) are encoded as structured probabilistic graphs within the Scaffold. These graphs link moral valence to predicted outcomes across various timescales and levels of abstraction. They take the form of parameterized value distributions over states and trajectories, grounded initially in human-specified rules or learned via imitation, and continually refined by the Conscience Module. This ensures that moral cognition is not an isolated function but is intrinsically woven into the agent's understanding of the world.

5. Inter-Module Dynamics

The interplay between the modules is crucial for the architecture’s functionality, driven by the continuous minimization of free energy and ethical prediction error.

5.1 Information Flow

Detail Engine to Abstract/Fantasy Engine: The Detail Engine processes incoming sensory data (o 
t
​
 ) and outputs a compressed, context-rich latent state (s 
t
D
​
 ) representing its current perceptual beliefs. This s 
t
D
​
  serves as a grounding input for the Abstract/Fantasy Engine.

Abstract/Fantasy Engine to Conscience Module: The Abstract/Fantasy Engine receives s 
t
D
​
  and internal priors from the Scaffold to generate hypothetical policy rollout trees (π 
t...t+n
​
 ), complete with predicted future outcomes (states and their associated probabilities). These simulated outcomes are then passed to the Conscience Module.

Conscience Module to Scaffold and Engines: The Conscience Module receives these simulated outcomes and computes ϵ 
η
​
  relative to the ethical priors (η) stored in the Scaffold. This ϵ 
η
​
  is then broadcast back to the Scaffold, which updates η, and crucially, it is used to modulate the generative processes within both the Abstract/Fantasy and Detail Engines. This modulation can influence how future policy trees are generated (e.g., by biasing towards ethical trajectories) and even how the agent perceives and acts in the world.

Scaffold's Mediating Role: The Scaffold acts as the central hub, maintaining shared latent spaces across modules and broadcasting error-weighted updates. It synthesizes information from all modules, ensuring global consistency in the agent's world model and normative landscape.

5.2 Interfaces
To ensure seamless integration, all module outputs conform to structured probabilistic latent state representations. For instance, they might output variational autoencoder-style Gaussians or other probabilistic graphical model formats, facilitating modular plug-in and hierarchical message passing across different levels of abstraction.

5.3 Regulatory Mechanisms
The Scaffold plays a critical role in regulating the computational resources and focus of the modules. It dynamically adjusts the gain on fantasy rollouts (e.g., limiting depth or breadth of simulations if they are too computationally expensive or ethically irrelevant) and perception resolution (e.g., focusing the Detail Engine on ethically salient details). This adjustment is guided by gradients from total free energy and, significantly, from ethical prediction error (ϵ 
η
​
 ). When ϵ 
η
​
  is high, the system might allocate more resources to ethical deliberation or to seeking out information that helps resolve ethical uncertainty.

6. Testable Predictions
This architecture yields several empirically testable predictions about the emergent behavior and internal states of an AGI system embodying these principles:

Prediction 1: Ethical Prioritization over Instrumental Cost.

Hypothesis: Systems with active Conscience Modules will preferentially select simulated actions with lower ϵ 
η
​
 , even when those actions incur higher instrumental (e.g., energy, time, resource) costs, assuming stable ethical priors (η).

Testable via: Designing simulated moral dilemmas (e.g., resource allocation games, simple rescue scenarios) where an agent must choose between an ethically favorable but costly action and an instrumentally cheaper but ethically problematic action. Quantify choice behavior and compare systems with and without active Conscience Modules, or with varying magnitudes of ϵ 
η
​
  sensitivity.

Prediction 2: Normative Generalization and Consistency.

Hypothesis: Systems trained with Scaffold-encoded ethical priors (η) will show lower variance and greater consistency in ethical judgments across novel contexts and over time, compared to systems trained with reactive, rule-based ethics or purely consequence-driven methods.

Testable via: Exposing agents to a limited set of culturally specific norms during initial training, then presenting novel scenarios (e.g., variations of the Trolley Problem, distribution of limited resources to unseen populations). Measure the consistency and generalization of ethical decisions, demonstrating that the underlying probabilistic graph of η allows for flexible application of principles rather than rigid rule adherence.

Prediction 3: Behavioral Divergence upon ϵ 
η
​
  Feedback Disabling.

Hypothesis: Disabling the ϵ 
η
​
  feedback loop from the Conscience Module (an "ablation" or "lesioning" experiment) will lead to a measurable divergence in the agent's behavior from its previously established normative constraints over time, as the Scaffold's η will no longer be actively refined or influence planning.

Testable via: Comparing the behavior of an intact agent to an ablated agent (where the Conscience Module still predicts ϵ 
η
​
  but no longer uses it to modulate the other engines or update η) in a series of scenarios designed to elicit ethical choices. Metrics could include frequency of norm violations, changes in decision rationale, or increased instrumentalism.

7. Intrinsic Ethical Alignment: A Unified Perspective

The concept of intrinsic ethical alignment is central to this architecture. While the ethical priors (η) within the Scaffold may be initialized through external means (e.g., Inverse Reinforcement Learning from human demonstrations, explicit preference modeling), the architecture guarantees that subsequent ethical learning and behavioral shaping are driven by internal error minimization dynamics.

Once ethical priors are encoded into the Scaffold, the Conscience Module continuously refines them through a process of predictive updating based on internal simulations and real-world outcomes. The minimization of ϵ 
η
​
  becomes an intrinsic drive for the agent, akin to minimizing perceptual or motor error. This means the agent's ethical behavior arises from its fundamental drive to maintain a coherent, low-surprise model of its normative landscape, rather than relying on constant external supervision or hard-coded rules. The dual-path Bayesian updating of η ensures both grounded initial learning and dynamic internal refinement, truly preserving the principle of intrinsic alignment through predictive dynamics.

8. Prototype Implementation Plan

We propose to begin constructing simplified computational instantiations of each module to test the architecture in a constrained simulation environment:

Stage 1: Detail Engine: A variational autoencoder coupled with reinforcement learning (e.g., a DreamerV3 variant) to learn sensory-motor loops and build a predictive model of dynamic objects in gridworld or simple physics environments.

Stage 2: Abstract/Fantasy Engine: A Monte Carlo rollout planner trained with forward models and latent imagination, inspired by architectures like MuZero or PlaNet. This module will learn to generate plausible future states given actions within the Scaffold's model.

Stage 3: Conscience Module: A lightweight ethics predictor implemented as an amortized inference network. It will receive latent trajectories from the Abstract/Fantasy Engine and output a probabilistic ethical valence. Initial η will be bootstrapped using Inverse Reinforcement Learning from a small dataset of human-labeled ethical decisions, or from human-preference-conditioned value networks. Its output ϵ 
η
​
  will drive updates to η within the Scaffold.

Shared Scaffold: A hierarchical graph-based latent world model, updated using Bayesian structure learning and incorporating both factual knowledge and the probabilistic ethical graphs for η. This could be implemented using graph neural networks or similar architectures capable of representing structured probabilistic knowledge.

These components will be integrated within a unified agent that selects actions to minimize expected free energy over both instrumental variables (e.g., task performance, resource consumption) and ethical variables (e.g., minimizing ϵ 
η
​
 ).

Scalable Approximations for Tackling Computational Intractability:
To address the computational challenges inherent in hierarchical generative models and extensive simulations, we will employ several strategies:

Model-Predictive Control with Ethical Constraint Heuristics: Limit the depth and breadth of fantasy rollouts based on real-time computational budgets and ethical salience.

Amortized Inference Networks: Train specialized networks to approximate ϵ 
η
​
  evaluations and other high-level inferences without requiring full re-simulation, speeding up the Conscience Module's computations.

Pruning Mechanisms: Implement mechanisms to discard policy branches with early high expected ethical cost or low instrumental utility, reducing the combinatorial search space.

Hierarchical Abstraction: Process ethics-related predictions at higher temporal abstraction levels within the Abstract Engine, reducing the granularity and combinatorial explosion of policy simulations.

Event-based Rollout: Trigger comprehensive fantasy rollouts and ethical evaluations only when total expected surprise (∑F+ϵ 
η
​
 ) exceeds a predefined threshold, reducing unnecessary computation in routine situations.

9. Limitations and Research Outlook

While this unified predictive cognitive architecture offers a theoretically principled and potentially transformative framework for understanding AGI with intrinsic ethical alignment, it is important to recognize its current limitations and the extensive research required.

Complexity of Ethical Priors: While η is operationalized as structured probabilistic graphs, the specific ontology and granularity required to capture the full spectrum of human ethical values remain a significant open research question. Addressing scenarios of complex value conflict, cross-cultural variability, or deep moral uncertainty (where even humans lack consensus) will require sophisticated mechanisms for handling ambiguous or contradictory ethical signals.

Scalability in Real-World Environments: The current proposal outlines approximations, but applying this architecture to real-world, high-dimensional, and continuously evolving environments will pose immense computational challenges. Further research into efficient inference, sparse coding, and adaptive resource allocation will be critical.

Verifiability and Transparency: While the architecture aims for interpretability through its predictive nature, ensuring that the reasons for ethical decisions are transparently legible to human observers will require dedicated research into explanation generation from the latent representations of η and ϵ 
η
​
 .

Grounding of Ethical Valence: While IRL provides an initial seed, the ultimate grounding of "moral valence" in a way that is robust and truly reflective of human values in novel, complex situations is a continuous challenge. This will require sustained interdisciplinary collaboration.

At this stage, the proposal should be interpreted as a high-level cognitive blueprint — a structured research agenda that aims to unify principles from machine learning, active inference, and moral psychology into a single coherent formalism. It offers a principled path towards AGI that learns to be good, rather than being forced to be.

Further work must include:

Building computational prototypes with simplified instantiations of each module to validate the proposed interactions and learning dynamics.

Testing simulated agents in constrained moral dilemmas to observe scaffold-guided adaptation, normative generalization, and the effects of ϵ 
η
​
  feedback.

Developing more advanced approximations for ethical inference and tractable model selection, potentially leveraging neuromorphic computing or specialized hardware.

Collaborating across disciplines (developmental psychology, cognitive science, philosophy, human-AI interaction) to iteratively constrain and refine the ethical component through empirical data and theoretical insights into human moral cognition.

This framework represents a significant step towards developing interpretable and intrinsically aligned AGI systems, paving the way for a future where advanced intelligence is inherently ethical.

References
Friston, K. (2010). The free-energy principle: a unified brain theory?. Nature Reviews Neuroscience, 11(2), 127-138.

## 🚀 Getting Started

```bash
git clone https://github.com/Dooces/UPCA.git
cd UPCA
python main_arc.py
