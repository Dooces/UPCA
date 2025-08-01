import pickle
import os
import numpy as np
import logging
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import cosine

class SIScaffold:
    """
    Scaffold/Integrative Memory (SI) for UPCA.
    Stores experience, priors, value estimates, learning progress, and parameterized transformations.
    Uses feature-based similarity for contextual recall.
    Tracks rule coverage, skill valence, and supports hierarchical skill composition.
    """

    def __init__(self, me_engine=None):
        self.experience = []
        self.value_table = {}
        self.priors = {}
        self.learning_log = []
        self.meta = {}
        self.transformations = []
        self.rule_coverage = {}
        self.logger = logging.getLogger("SIScaffold")
        self.skill_counter = 0
        self.me_engine = me_engine

    def store_experience(self, state, action, reward, next_state):
        self.experience.append((state, action, reward, next_state))
        self.logger.debug(f"SI: Stored experience: {action}")

    def update_value(self, state, action, value):
        key = (self._state_repr(state), self._action_repr(action))
        self.value_table[key] = value

    def get_value(self, state, action):
        key = (self._state_repr(state), self._action_repr(action))
        return self.value_table.get(key, 0.0)

    def update_priors(self, key, value):
        self.priors[key] = value

    def get_prior(self, key):
        return self.priors.get(key, None)

    def log_learning(self, step, progress, success, valence):
        entry = {
            "step": step,
            "progress": progress,
            "success": success,
            "valence": valence
        }
        self.learning_log.append(entry)
        self.logger.debug(f"SI: Learning log entry: {entry}")

    def _state_repr(self, state):
        if isinstance(state, (list, tuple)):
            return tuple(map(tuple, state))
        elif hasattr(state, 'numpy'):
            return tuple(map(tuple, state.numpy()))
        elif isinstance(state, str):
            return state
        else:
            return str(state)

    def _action_repr(self, action):
        if isinstance(action, list):
            return tuple((name, tuple(sorted(params.items()))) for name, params in action)
        elif callable(action):
            return action.__name__ if hasattr(action, '__name__') else str(action)
        elif isinstance(action, (list, tuple)):
            return tuple(action)
        else:
            return str(action)

    def get_state_for_ME(self):
        return {"value_table": self.value_table, "priors": self.priors}

    def get_state_for_MA(self):
        return {"experience": self.experience}

    def get_state_for_AMC(self):
        return {"learning_log": self.learning_log, "priors": self.priors}

    def _extract_features(self, grid):
        """Converts a raw grid into a conceptual feature vector."""
        # Handle various input types
        if isinstance(grid, list):
            # If it's a list of objects, use the first one or flatten
            if len(grid) == 0:
                grid = np.zeros((1, 1), dtype=int)
            elif isinstance(grid[0], (np.ndarray, list)):
                grid = np.array(grid[0])
            else:
                grid = np.array(grid)
        elif not hasattr(grid, "shape"):
            grid = np.array(grid)
        
        # Ensure 2D
        if grid.ndim != 2:
            if grid.size == 0:
                grid = np.zeros((1, 1), dtype=int)
            else:
                # Try to make it square-ish
                side = int(np.sqrt(grid.size))
                if side * side == grid.size:
                    grid = grid.reshape((side, side))
                else:
                    # Make it a column vector
                    grid = grid.reshape((-1, 1))
        
        features = []
        height, width = grid.shape
        num_pixels = np.sum(grid > 0)
        unique_colors = np.unique(grid[grid > 0]) if num_pixels > 0 else []
        num_colors = len(unique_colors)
        features.extend([height, width, num_pixels, num_colors])

        # Color histogram (fixed size)
        color_hist = np.zeros(10)
        if num_pixels > 0:
            colors, counts = np.unique(grid[grid > 0], return_counts=True)
            for c, count in zip(colors, counts):
                if 0 < c < 10: 
                    color_hist[int(c)] = count
        features.extend(color_hist.tolist())

        # Object-based features (top 3 objects)
        try:
            labeled_array, num_objects = label(grid > 0)
            features.append(num_objects)
            
            if num_objects > 0:
                object_slices = find_objects(labeled_array)
                object_sizes = []
                for i in range(num_objects):
                    obj_size = np.sum(labeled_array == (i + 1))
                    if obj_size > 0:
                        object_sizes.append((obj_size, object_slices[i]))
                
                sorted_objects = sorted(object_sizes, key=lambda x: x[0], reverse=True)
                
                for i in range(3):
                    if i < len(sorted_objects):
                        size, slc = sorted_objects[i]
                        obj_grid = grid[slc]
                        obj_height, obj_width = obj_grid.shape
                        center_y = (slc[0].start + slc[0].stop) / 2 / max(height, 1)
                        center_x = (slc[1].start + slc[1].stop) / 2 / max(width, 1)
                        features.extend([size, obj_height, obj_width, center_y, center_x])
                    else:
                        features.extend([0, 0, 0, 0, 0])
                
                # Relational feature: vector from largest to second largest object
                if len(sorted_objects) > 1:
                    _, slc1 = sorted_objects[0]
                    _, slc2 = sorted_objects[1]
                    center1_y = (slc1[0].start + slc1[0].stop) / 2
                    center1_x = (slc1[1].start + slc1[1].stop) / 2
                    center2_y = (slc2[0].start + slc2[0].stop) / 2
                    center2_x = (slc2[1].start + slc2[1].stop) / 2
                    features.extend([center2_y - center1_y, center2_x - center1_x])
                else:
                    features.extend([0, 0])
            else:
                features.append(0)  # num_objects
                features.extend([0] * 17)  # 3 objects * 5 features + 2 relational
        except Exception as e:
            self.logger.debug(f"Error in object extraction: {e}")
            features.append(0)  # num_objects
            features.extend([0] * 17)

        # Symmetry features
        try:
            v_symmetric = float(np.array_equal(grid, np.flip(grid, axis=0)))
            h_symmetric = float(np.array_equal(grid, np.flip(grid, axis=1)))
            features.extend([v_symmetric, h_symmetric])
        except:
            features.extend([0, 0])

        # Pattern repetition features
        try:
            if height > 1 and width > 1:
                row_repetition = np.mean([np.array_equal(grid[i], grid[i+1]) for i in range(height-1)])
                col_repetition = np.mean([np.array_equal(grid[:,i], grid[:,i+1]) for i in range(width-1)])
                features.extend([row_repetition, col_repetition])
            else:
                features.extend([0, 0])
        except:
            features.extend([0, 0])

        # Connectivity features: number of connected components per color (top 5 colors)
        connectivity_features = []
        try:
            for color in range(1, min(6, int(np.max(grid))+1) if num_pixels > 0 else 1):
                color_mask = (grid == color)
                if np.any(color_mask):
                    labeled_color, num = label(color_mask)
                    connectivity_features.append(num)
                else:
                    connectivity_features.append(0)
        except:
            pass
        
        # Ensure we have exactly 5 connectivity features
        while len(connectivity_features) < 5:
            connectivity_features.append(0)
        features.extend(connectivity_features[:5])

        # Pad to fixed length
        target_feature_length = 50
        if len(features) < target_feature_length:
            features.extend([0] * (target_feature_length - len(features)))
        
        return np.array(features[:target_feature_length], dtype=np.float32)

    def robust_similarity_fn(self, grid1, grid2):
        """Calculates similarity based on conceptual features, not raw pixels."""
        try:
            # Convert to numpy arrays if needed
            grid1 = np.array(grid1) if not isinstance(grid1, np.ndarray) else grid1
            grid2 = np.array(grid2) if not isinstance(grid2, np.ndarray) else grid2
            
            if np.array_equal(grid1, grid2): 
                return 1.0
            
            f1 = self._extract_features(grid1)
            f2 = self._extract_features(grid2)
            
            f1_norm = np.linalg.norm(f1)
            f2_norm = np.linalg.norm(f2)
            
            if f1_norm == 0 or f2_norm == 0:
                return 0.0
            
            f1_normalized = f1 / f1_norm
            f2_normalized = f2 / f2_norm
            
            # Compute cosine similarity
            similarity = 1 - cosine(f1_normalized, f2_normalized)
            
            # Ensure similarity is in [0, 1]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            self.logger.debug(f"Error in similarity computation: {e}")
            return 0.0

    def store_transformation(self, input_grid, action_sequence, output_grid, generalize=True):
        try:
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            
            if np.array_equal(input_array, output_array) or np.all(input_array == 0) or np.all(output_array == 0):
                self.logger.debug("SI: Rejected trivial transformation.")
                return
            
            self.skill_counter += 1
            skill_name = f"SKILL_{self.skill_counter}"
            features = self._extract_features(input_array)
            
            self.transformations.append({
                "input_pattern": input_array.copy(),
                "action_sequence": action_sequence,
                "output_pattern": output_array.copy(),
                "generalize": generalize,
                "skill_name": skill_name,
                "features": features,
                "valence_history": [],
                "avg_valence": 0.5
            })
            
            # Update rule coverage
            for item in action_sequence:
                if isinstance(item, tuple) and len(item) == 2:
                    name, params = item
                    if isinstance(params, dict):
                        key = (name, tuple(sorted(params.items())))
                    else:
                        key = (name, ())
                else:
                    key = (str(item), ())
                
                if key not in self.rule_coverage:
                    self.rule_coverage[key] = set()
                self.rule_coverage[key].add(self._state_repr(input_array))
                self.logger.debug(f"SI: Rule {key} now covers {len(self.rule_coverage[key])} patterns")
            
            # Check for promotion to general rule
            for item in action_sequence:
                if isinstance(item, tuple) and len(item) == 2:
                    name, params = item
                    if isinstance(params, dict):
                        key = (name, tuple(sorted(params.items())))
                    else:
                        key = (name, ())
                else:
                    key = (str(item), ())
                
                if len(self.rule_coverage.get(key, set())) > 10:
                    for trans in self.transformations:
                        if trans["action_sequence"] == action_sequence and not trans["generalize"]:
                            trans["generalize"] = True
                            self.logger.info(f"SI: Promoted to abstract rule: {key}")
        except Exception as e:
            self.logger.error(f"Error storing transformation: {e}")

    def find_relevant_skills(self, grid, top_k=5, valence_weight=0.3):
        """
        Return the top_k most relevant skills for the current grid, based on feature similarity and valence.
        """
        scored_skills = []
        for trans in self.transformations:
            if not trans.get("generalize", False):
                continue
            try:
                pattern = trans["input_pattern"]
                similarity_score = self.robust_similarity_fn(pattern, grid)
                valence_score = trans.get("avg_valence", 0.5)  # Default neutral
                combined_score = (1 - valence_weight) * similarity_score + valence_weight * valence_score
                scored_skills.append((combined_score, trans))
            except Exception as e:
                self.logger.debug(f"Error scoring skill: {e}")
                continue
        
        scored_skills.sort(reverse=True, key=lambda x: x[0])
        return [trans for score, trans in scored_skills[:top_k]]

    def find_general_skill(self, skill_name="any"):
        if skill_name == "any":
            return [trans for trans in self.transformations if trans.get("generalize", False)]
        return [trans for trans in self.transformations
                if trans.get("generalize", False) and 
                (trans.get("skill_name") == skill_name or 
                 any(name == skill_name for name, _ in trans.get("action_sequence", [])))]

    def get_rule_coverage(self, skill_name, params):
        if isinstance(params, dict):
            key = (skill_name, tuple(sorted(params.items())))
        else:
            key = (skill_name, ())
        return len(self.rule_coverage.get(key, set()))

    def get_skill_by_name(self, skill_name):
        for trans in self.transformations:
            if trans.get("skill_name") == skill_name:
                return trans
        return None

    def compose_skills(self, skill1_name, skill2_name):
        """Compose two skills to create a new compound skill."""
        skill1 = self.get_skill_by_name(skill1_name)
        skill2 = self.get_skill_by_name(skill2_name)
        if skill1 and skill2:
            composed_sequence = skill1["action_sequence"] + skill2["action_sequence"]
            self.skill_counter += 1

            composed_name = f"COMPOSED_{self.skill_counter}"
            self.transformations.append({
                "skill_name": composed_name,
                "action_sequence": composed_sequence,
                "component_skills": [skill1_name, skill2_name],
                "generalize": True,
                "features": None  # Will be computed when applied
            })
            return composed_name
        return None

    def update_skill_valence(self, skill_name, valence_score):
        """Update the ethical/success valence of a skill."""
        for trans in self.transformations:
            if trans.get("skill_name") == skill_name:
                if "valence_history" not in trans:
                    trans["valence_history"] = []
                trans["valence_history"].append(valence_score)
                trans["avg_valence"] = np.mean(trans["valence_history"])
                break

    def track_learning_progress(self):
        """Compute learning progress metrics for meta-learning."""
        if len(self.learning_log) < 2:
            return 0.0
        recent_window = self.learning_log[-10:]
        older_window = self.learning_log[-20:-10] if len(self.learning_log) >= 20 else self.learning_log[:10]
        recent_success = np.mean([entry["success"] for entry in recent_window])
        older_success = np.mean([entry["success"] for entry in older_window])
        learning_progress = recent_success - older_success
        return learning_progress

    def save(self, filename="si_scaffold.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_or_create(cls, filename="si_scaffold.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            return cls()
