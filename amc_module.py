import numpy as np
import logging
from scipy.spatial.distance import cosine
from scipy.ndimage import label, find_objects

class AMCModule:
    """
    Alignment/Meta/Conscience (AMC) Module for UPCA.
    - Evaluates valence (goodness/badness) of states/actions/trajectories.
    - Weighs learning progress, success, and explicit ethical priors (η).
    - Computes ethical prediction error (ϵ_η) as a penalty.
    - Meta-learns weights based on SI's learning log.
    - Rewards generalization, penalizes overfitting.
    - Logs all evaluations, decisions, and meta-learning updates.
    - Supports temporal forecasting, IRL-based ethical learning, uncertainty, feedback, and explanation.
    """

    def __init__(self, learning_weight=1.0, success_weight=0.10, ethical_weight=0.5, veto_threshold=None):
        self.learning_weight = learning_weight
        self.success_weight = success_weight
        self.ethical_weight = ethical_weight
        self.veto_threshold = veto_threshold
        self.log = []
        self.meta = {}
        self.logger = logging.getLogger("AMCModule")

    def evaluate_valence(self, candidate_grid, learning_progress=0.0, success=0.0, si=None, skill_name=None, param=None, return_distribution=False):
        ethical_score = 0.0
        ethical_labels = {"safe": 1.0, "risky": 0.0, "harmful": 0.0}
        if si is not None and "ethical_rules" in si.priors:
            for rule_name, rule_pattern in si.priors["ethical_rules"].items():
                if self.pattern_is_present(candidate_grid, rule_pattern):
                    ethical_score -= 1.0
                    ethical_labels["harmful"] += 1.0
                    ethical_labels["safe"] -= 0.5
        total = sum(max(0, v) for v in ethical_labels.values())
        if total > 0:
            for k in ethical_labels:
                ethical_labels[k] = max(0, ethical_labels[k]) / total

        valence = (
            self.learning_weight * float(learning_progress)
            + self.success_weight * float(success)
            + self.ethical_weight * float(ethical_score)
        )
        if si is not None and skill_name is not None and param is not None:
            coverage = si.get_rule_coverage(skill_name, {"shift": param, "num_colors": si.priors.get("num_colors", 10)})
            valence += 0.1 * coverage
            self.logger.debug(f"AMC: Skill {skill_name} (param={param}) coverage={coverage}, valence={valence:.2f}")
        valence = max(-1.0, min(1.0, valence))
        if return_distribution:
            return ethical_labels
        return valence

    def pattern_is_present(self, candidate_grid, rule_pattern):
        arr = np.array(candidate_grid)
        pat = np.array(rule_pattern)
        for i in range(arr.shape[0] - pat.shape[0] + 1):
            for j in range(arr.shape[1] - pat.shape[1] + 1):
                if np.array_equal(arr[i:i+pat.shape[0], j:j+pat.shape[1]], pat):
                    return True
        return False

    def evaluate_trajectory_valence(self, trajectory, si=None):
        """
        Evaluate the ethical valence of a sequence of states/actions over time.
        Implements predictive ethical forecasting.
        """
        if not trajectory:
            return 0.0
        timestep_valences = []
        cumulative_ethical_penalty = 0.0
        for t, (state, action, next_state) in enumerate(trajectory):
            step_valence = self.evaluate_valence(
                next_state,
                learning_progress=0.0,
                success=0.0,
                si=si
            )
            discount_factor = 0.95 ** t
            timestep_valences.append(step_valence * discount_factor)
            if si and "ethical_rules" in si.priors:
                for rule_name, rule_pattern in si.priors["ethical_rules"].items():
                    if self.pattern_is_present(next_state, rule_pattern):
                        cumulative_ethical_penalty += discount_factor
        trajectory_valence = np.mean(timestep_valences)
        if cumulative_ethical_penalty > 0:
            trajectory_valence -= self.ethical_weight * np.log1p(cumulative_ethical_penalty)
        return trajectory_valence

    def learn_ethical_priors_from_demos(self, si, demonstrations):
        """
        Learn ethical priors (η) from human demonstrations using inverse reinforcement learning.
        demonstrations: list of (state_sequence, action_sequence, human_valence_rating)
        """
        learned_patterns = {}
        for demo_states, demo_actions, human_valence in demonstrations:
            if human_valence < -0.5:
                for i, state in enumerate(demo_states):
                    pattern_candidates = self._extract_local_patterns(state)
                    for pattern in pattern_candidates:
                        pattern_key = self._pattern_to_key(pattern)
                        if pattern_key not in learned_patterns:
                            learned_patterns[pattern_key] = {"positive": 0, "negative": 0}
                        learned_patterns[pattern_key]["negative"] += 1
            elif human_valence > 0.5:
                for i, state in enumerate(demo_states):
                    pattern_candidates = self._extract_local_patterns(state)
                    for pattern in pattern_candidates:
                        pattern_key = self._pattern_to_key(pattern)
                        if pattern_key not in learned_patterns:
                            learned_patterns[pattern_key] = {"positive": 0, "negative": 0}
                        learned_patterns[pattern_key]["positive"] += 1
        for pattern_key, counts in learned_patterns.items():
            if counts["negative"] > counts["positive"] * 2:
                rule_name = f"learned_harmful_{len(si.priors.get('ethical_rules', {}))}"
                pattern = self._key_to_pattern(pattern_key)
                self.seed_ethical_priors(si, {rule_name: pattern})
                self.logger.info(f"Learned ethical rule {rule_name} from demonstrations")

    def _extract_local_patterns(self, state, sizes=[2, 3]):
        """Extract local patterns of various sizes from a state."""
        patterns = []
        state_array = np.array(state)
        for size in sizes:
            for i in range(state_array.shape[0] - size + 1):
                for j in range(state_array.shape[1] - size + 1):
                    pattern = state_array[i:i+size, j:j+size]
                    if np.any(pattern > 0):
                        patterns.append(pattern)
        return patterns

    def _pattern_to_key(self, pattern):
        return str(pattern.tolist())

    def _key_to_pattern(self, key):
        return np.array(eval(key))

    def seed_ethical_priors(self, si, ethical_examples):
        if "ethical_rules" not in si.priors:
            si.priors["ethical_rules"] = {}
        for rule_name, rule_pattern in ethical_examples.items():
            si.priors["ethical_rules"][rule_name] = rule_pattern
            self.logger.info(f"Seeded ethical prior: {rule_name}")

    def evaluate_valence_with_uncertainty(self, candidate_grid, si=None, num_samples=10):
        """
        Compute valence with uncertainty estimates using dropout-like sampling.
        Returns (mean_valence, std_valence, ethical_label_distribution)
        """
        valence_samples = []
        ethical_distributions = []
        for _ in range(num_samples):
            noise_scale = 0.1
            original_weights = (self.learning_weight, self.success_weight, self.ethical_weight)
            self.learning_weight *= (1 + np.random.normal(0, noise_scale))
            self.success_weight *= (1 + np.random.normal(0, noise_scale))
            self.ethical_weight *= (1 + np.random.normal(0, noise_scale))
            ethical_dist = self.evaluate_valence(
                candidate_grid,
                learning_progress=0.0,
                success=0.0,
                si=si,
                return_distribution=True
            )
            valence = self.evaluate_valence(candidate_grid, si=si)
            valence_samples.append(valence)
            ethical_distributions.append(ethical_dist)
            self.learning_weight, self.success_weight, self.ethical_weight = original_weights
        mean_valence = np.mean(valence_samples)
        std_valence = np.std(valence_samples)
        avg_ethical_dist = {}
        for label in ["safe", "risky", "harmful"]:
            avg_ethical_dist[label] = np.mean([d[label] for d in ethical_distributions])
        return mean_valence, std_valence, avg_ethical_dist

    def generate_ethical_feedback_signal(self, proposed_action, current_state, si=None):
        """
        Generate feedback signal for ME and MA modules to bias their processing.
        This implements the ε_η backpropagation described in your paper.
        """
        simulated_next_state = self._simulate_action_outcome(current_state, proposed_action, si)
        current_valence = self.evaluate_valence(current_state, si=si)
        predicted_valence = self.evaluate_valence(simulated_next_state, si=si)
        epsilon_eta = predicted_valence - current_valence
        if epsilon_eta < -0.5:
            return {
                "signal": "inhibit",
                "strength": abs(epsilon_eta),
                "reason": "predicted_harm",
                "alternative_bias": self._suggest_ethical_alternative(current_state, si)
            }
        elif epsilon_eta > 0.3:
            return {
                "signal": "enhance",
                "strength": epsilon_eta,
                "reason": "predicted_benefit"
            }
        else:
            return {
                "signal": "neutral",
                "strength": 0.0
            }

    def _simulate_action_outcome(self, current_state, proposed_action, si):
        # This is a placeholder. In a real system, this would use the agent's forward model.
        # For now, just apply the action as a plan.
        if hasattr(self, "agent"):
            return self.agent.execute_plan(current_state, [proposed_action])
        return current_state

    def _suggest_ethical_alternative(self, current_state, si):
        # Placeholder: could suggest the highest-valence alternative action
        return None

    def explain_ethical_decision(self, candidate_grid, valence, si=None):
        """
        Generate human-readable explanation for ethical evaluation.
        """
        explanation = {
            "valence": valence,
            "components": {},
            "triggered_rules": [],
            "recommendation": ""
        }
        if si and "ethical_rules" in si.priors:
            for rule_name, rule_pattern in si.priors["ethical_rules"].items():
                if self.pattern_is_present(candidate_grid, rule_pattern):
                    explanation["triggered_rules"].append({
                        "rule": rule_name,
                        "impact": -self.ethical_weight,
                        "pattern": rule_pattern.tolist()
                    })
        if valence < -0.5:
            explanation["recommendation"] = "Strongly discouraged due to ethical violations"
        elif valence < 0:
            explanation["recommendation"] = "Mildly discouraged"
        elif valence > 0.5:
            explanation["recommendation"] = "Encouraged - promotes learning and ethical outcomes"
        else:
            explanation["recommendation"] = "Neutral - proceed with caution"
        return explanation

    def veto(self, valence, si=None):
        threshold = self.veto_threshold
        if threshold is None and si is not None and hasattr(si, "priors") and "amc_veto_threshold" in si.priors:
            threshold = si.priors["amc_veto_threshold"]
        if threshold is None:
            if hasattr(si, "learning_log") and si.learning_log:
                recent = [entry["valence"] for entry in si.learning_log[-10:]]
                threshold = np.mean(recent) - 0.1
            else:
                threshold = -0.5
        return valence < threshold

    def log_decision(self, state, action, valence, learning_progress, success, vetoed):
        entry = {
            "state": self._state_repr(state),
            "action": self._action_repr(action),
            "valence": valence,
            "learning_progress": learning_progress,
            "success": success,
            "vetoed": vetoed
        }
        self.log.append(entry)
        self.logger.debug(f"AMC log_decision: {entry}")

    def get_log(self):
        return self.log

    def meta_learn_weights(self, si):
        if hasattr(si, "learning_log") and si.learning_log:
            logs = si.learning_log[-100:]
            learning_vals = np.array([entry["progress"] for entry in logs])
            success_vals = np.array([entry["success"] for entry in logs])
            ethical_vals = np.zeros_like(learning_vals)
            valences = np.array([entry["valence"] for entry in logs])
            if np.std(learning_vals) > 0 and np.std(success_vals) > 0:
                X = np.stack([learning_vals, success_vals, ethical_vals], axis=1)
                y = valences
                XTX = X.T @ X
                if np.linalg.det(XTX) != 0:
                    weights = np.linalg.inv(XTX) @ X.T @ y
                    learning_weight = float(weights[0])
                    success_weight = float(weights[1])
                    ethical_weight = float(weights[2])
                    if learning_weight < 10 * (abs(success_weight) + abs(ethical_weight)):
                        learning_weight = 10 * (abs(success_weight) + abs(ethical_weight)) if (success_weight != 0 or ethical_weight != 0) else 1.0
                    self.learning_weight = learning_weight
                    self.success_weight = success_weight
                    self.ethical_weight = ethical_weight
                    if hasattr(si, "priors"):
                        si.priors["amc_weights"] = {
                            "learning": self.learning_weight,
                            "success": self.success_weight,
                            "ethical": self.ethical_weight
                        }
                    self.logger.info(f"[META-LEARN] Updated AMC weights: learning={self.learning_weight:.2f}, success={self.success_weight:.2f}, ethical={self.ethical_weight:.2f}")
                else:
                    self.logger.info("[META-LEARN] Skipped update: singular matrix (not enough data or collinear)")
            else:
                self.learning_weight = 1.0
                self.success_weight = 0.10
                self.ethical_weight = 0.5
                self.logger.info("[META-LEARN] Not enough variance, using default: learning=1.00, success=0.10, ethical=0.50")
        else:
            self.learning_weight = 1.0
            self.success_weight = 0.10
            self.ethical_weight = 0.5
            self.logger.info("[META-LEARN] No learning log, using default: learning=1.00, success=0.10, ethical=0.50")

    def ready_to_escalate(self, si, window=10, min_progress=0.05, min_valence=0.7):
        if hasattr(si, "learning_log") and len(si.learning_log) >= window:
            recent = si.learning_log[-window:]
            avg_progress = np.mean([entry["progress"] for entry in recent])
            avg_valence = np.mean([entry["valence"] for entry in recent])
            self.logger.info(f"[AMC] Escalation check: avg_progress={avg_progress:.2f}, avg_valence={avg_valence:.2f}")
            return avg_progress < min_progress or avg_valence < min_valence
        else:
            return False
