import numpy as np
import logging
import random
from queue import PriorityQueue
import itertools

class UPCAAgent:
    """
    Unified Predictive Cognitive Architecture (UPCA) Agent.
    Orchestrates ME (Model Executor), MA (Model Actor), AMC (Conscience), and SI (Scaffold).
    Implements a cognitive cycle: fast ME recall, slow MA planning, AMC arbitration, SI memory.
    Plans are represented hierarchically: primitive actions and named skills.
    """

    DREAMING_ENABLED = True

    def __init__(self, me, ma, amc, si, log_level=logging.INFO, me_enabled=True, ma_enabled=True, me_efficiency=1.0, ma_efficiency=1.0, me_confidence=0.95):
        self.me = me
        self.ma = ma
        self.amc = amc
        self.si = si
        self.me_enabled = me_enabled
        self.ma_enabled = ma_enabled
        self.me_efficiency = me_efficiency
        self.ma_efficiency = ma_efficiency
        self.me_confidence = me_confidence
        self.logger = logging.getLogger("UPCAAgent")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        self.skill_counter = 0

    def _ensure_2d_array(self, grid):
        """Convert various input types to proper 2D numpy array."""
        if isinstance(grid, np.ndarray):
            if grid.ndim == 2:
                return grid
            elif grid.ndim == 1:
                # Try to make it square if possible
                size = int(np.sqrt(grid.size))
                if size * size == grid.size:
                    return grid.reshape((size, size))
                else:
                    return grid.reshape((-1, 1))
            elif grid.ndim > 2:
                return grid[0] if grid.shape[0] > 0 else np.zeros((1, 1), dtype=int)
        elif isinstance(grid, list):
            # Try to convert to numpy array
            try:
                arr = np.array(grid, dtype=int)
                return self._ensure_2d_array(arr)
            except:
                # If it fails, it might be a ragged array
                if len(grid) == 0:
                    return np.zeros((1, 1), dtype=int)
                # Try to extract first valid 2D grid
                if isinstance(grid[0], (list, np.ndarray)):
                    try:
                        return np.array(grid[0], dtype=int)
                    except:
                        pass
                # Last resort: create a column vector
                return np.array([[x] for x in grid], dtype=int)
        else:
            # Convert to array and try again
            try:
                return self._ensure_2d_array(np.array(grid))
            except:
                return np.zeros((1, 1), dtype=int)

    def observe_example(self, input_grid, output_grid):
        """Store an input-output example in the scaffold."""
        # Ensure grids are proper 2D arrays
        input_array = self._ensure_2d_array(input_grid)
        output_array = self._ensure_2d_array(output_grid)
        
        self.si.store_experience(input_array, None, 0.0, output_array)
        self.logger.debug(f"Observed example stored in SI. Input shape: {input_array.shape}, Output shape: {output_array.shape}")

    def execute_plan(self, grid, plan, depth=0, max_depth=20):
        """Executes a hierarchical plan, resolving skill names recursively."""
        if depth > max_depth:
            self.logger.warning("Execution failed: plan recursion too deep (possible cycle).")
            return grid
        
        current_grid = self._ensure_2d_array(grid)
        
        for item in plan:
            if isinstance(item, tuple) and len(item) == 2:
                action_name, params = item
            else:
                action_name = str(item)
                params = {}
            
            # Check if it's a primitive action in ME
            if action_name in self.me.skills:
                current_grid = self.me.execute(current_grid, action_name, params)
            # Check if it's a learned skill
            elif action_name.startswith("SKILL_") or action_name.startswith("COMPOSED_"):
                skill_definition = self.si.get_skill_by_name(action_name)
                if skill_definition:
                    current_grid = self.execute_plan(current_grid, skill_definition['action_sequence'], depth=depth+1, max_depth=max_depth)
                else:
                    self.logger.warning(f"Execution failed: Unknown skill name '{action_name}'.")
            # Check if it's a learned rule
            elif action_name == 'learned_rule' and isinstance(params, dict) and 'operations' in params:
                current_grid = self.ma.apply_learned_rule(current_grid, params)
            else:
                # Try to find it in MA transformations
                transform_fn = self.ma.get_transformation_by_name(action_name)
                if transform_fn:
                    current_grid = self.apply_primitive_action(current_grid, transform_fn, params)
                else:
                    self.logger.warning(f"Execution failed: Unknown action/skill '{action_name}'.")
        
        return current_grid

    def apply_primitive_action(self, grid, action_fn, params=None):
        """Apply a primitive action with parameters."""
        if params is None:
            params = {}
        
        grid = self._ensure_2d_array(grid)
        
        # Handle string action names
        if isinstance(action_fn, str):
            if action_fn in self.me.skills:
                return self.me.execute(grid, action_fn, params)
            else:
                transform_fn = self.ma.get_transformation_by_name(action_fn)
                if transform_fn:
                    action_fn = transform_fn
                else:
                    self.logger.warning(f"Unknown action: {action_fn}")
                    return grid
        
        # Get function name
        fn_name = action_fn.__name__ if hasattr(action_fn, '__name__') else str(action_fn)
        
        # Apply the action based on its type
        if fn_name == "color_shift":
            return action_fn(grid, **params) if params else action_fn(grid)
        elif fn_name == "object_fill":
            return action_fn(grid, **params) if params else action_fn(grid)
        elif fn_name == "color_map":
            mapping = params.get("mapping")
            if mapping is None:
                self.logger.warning("color_map called without 'mapping' param. Returning grid unchanged.")
                return grid
            return action_fn(grid, mapping=mapping)
        elif fn_name == "flood_fill":
            start_pos = params.get("start_pos", (0, 0))
            new_color = params.get("new_color", 1)
            return action_fn(grid, start_pos=start_pos, new_color=new_color)
        elif fn_name == "compose_objects":
            if 'extract_objects' in self.me.skills:
                objects = self.me.skills['extract_objects'](grid)
                return action_fn(grid, objects)
            else:
                return action_fn(grid)
        elif hasattr(self.ma, "parameterized") and fn_name in self.ma.parameterized:
            return action_fn(grid, **params)
        else:
            return action_fn(grid)

    def solve_arc(self, input_grid, target_grid, examples=None, beam_width=3, max_steps=10):
        """
        Main solving algorithm for ARC tasks using A* search with learned rules and primitive actions.
        This method attempts to transform the `input_grid` into the `target_grid` by searching for a sequence of actions
        that maximize similarity to the target. It leverages learned rules from examples, creative "fantasy" search,
        primitive actions, and previously stored general skills. The search is guided by a cost function and similarity metric.
        Args:
            input_grid (array-like): The starting grid to be transformed.
            target_grid (array-like): The desired target grid.
            examples (list of tuple, optional): List of (input, output) example pairs for learning transformations.
            beam_width (int, optional): Number of best children to expand at each search step (default: 3).
            max_steps (int, optional): Maximum number of steps/actions in a solution path (default: 10).
        Returns:
            tuple:
                - result_grid (array-like): The resulting grid after applying the found transformation sequence.
                - path (list): List of (action_name, params) tuples representing the transformation sequence.
                - valence (float): The final evaluation score (e.g., similarity or reward) of the result.
        Notes:
            - If a high-confidence learned rule is found from examples, it is applied first.
            - The search explores learned rules, creative fantasy search, primitive actions, and known general skills.
            - The method returns the best found solution if a perfect match is not achieved.
        """
        """Main solving algorithm using A* search with learned rules and primitive actions."""
        ACTION_COSTS = {
            'identity': 0.1,
            'rotate90': 1.0,
            'flip_horizontal': 1.0,
            'flip_vertical': 1.0,
            'mirror_diagonal': 1.0,
            'largest_object_crop': 1.5,
            'crop_nonzero': 1.5,
            'erase_small_objects': 2.0,
            'object_fill': 5.0,
            'color_shift': 8.0,
            'learned_rule': 0.5,
            'rotate': 1.0,
            'flip': 1.0,
            'color_map': 2.0,
            'remove_objects': 3.0,
            'extract_pattern': 2.0,
            'repeat_pattern': 2.0,
            'flood_fill': 3.0,
            'symmetrize': 2.0,
        }

        # Ensure inputs are proper 2D arrays
        input_grid = self._ensure_2d_array(input_grid)
        target_grid = self._ensure_2d_array(target_grid)

        def grid_repr(grid):
            return str(np.array(grid).tolist())

        # Try to learn from examples if available
        learned_rule = None
        if examples and self.ma_enabled:
            try:
                # Ensure examples are proper 2D arrays
                clean_examples = []
                for ex_in, ex_out in examples:
                    clean_examples.append((self._ensure_2d_array(ex_in), self._ensure_2d_array(ex_out)))
                
                learned_rule = self.ma.learn_transformation_from_examples(clean_examples, self.si)
                if learned_rule and learned_rule['confidence'] > 0.7:
                    try:
                        result = self.ma.apply_learned_rule(input_grid, learned_rule)
                        sim = self.si.robust_similarity_fn(result, target_grid)
                        if sim > 0.95:
                            self.logger.info(f"Agent found solution using learned rule (confidence={learned_rule['confidence']:.2f}, sim={sim:.2f})")
                            return result, [('learned_rule', learned_rule)], 1.0
                    except Exception as e:
                        self.logger.warning(f"Failed to apply learned rule: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to learn from examples: {e}")

        q = PriorityQueue()
        counter = itertools.count()
        q.put((0 - 1.0, 0, next(counter), (input_grid, [], 0.0)))
        visited = set()
        best_found = (input_grid, [], -1.0)

        while not q.empty():
            priority, path_len, _, (current_grid, path, g_cost) = q.get()
            
            # Ensure current_grid is numpy array
            current_grid = self._ensure_2d_array(current_grid)
            
            sim = self.si.robust_similarity_fn(current_grid, target_grid)
            valence = self.amc.evaluate_valence(current_grid, sim, 0.0, si=self.si)
            self.logger.debug(f"A* Search: path_len={path_len}, sim={sim:.2f}, valence={valence:.2f}")

            if sim > 0.95:
                self.logger.info(f"Agent found solution: {path} (sim={sim:.2f}, valence={valence:.2f})")
                if path:
                    self.si.store_transformation(input_grid, path, current_grid, generalize=True)
                return current_grid, path, valence

            if sim > self.si.robust_similarity_fn(best_found[0], target_grid):
                best_found = (current_grid, path, valence)

            grid_key = grid_repr(current_grid)
            if grid_key in visited:
                continue
            visited.add(grid_key)

            children = []

            # 1. Try MA's learned rules if available
            if self.ma_enabled and learned_rule and learned_rule['confidence'] > 0.5:
                try:
                    next_grid = self.ma.apply_learned_rule(current_grid, learned_rule)
                    new_path = path + [('learned_rule', learned_rule)]
                    new_g_cost = g_cost + ACTION_COSTS.get('learned_rule', 0.5)
                    next_sim = self.si.robust_similarity_fn(next_grid, target_grid)
                    f_n = new_g_cost + (1.0 - next_sim)
                    children.append((f_n, len(new_path), next(counter), (next_grid, new_path, new_g_cost)))
                except:
                    pass

            # 2. Try MA's fantasy search for creative solutions
            if self.ma_enabled and examples and len(path) < max_steps // 2:
                try:
                    clean_examples = []
                    for ex_in, ex_out in examples:
                        clean_examples.append((self._ensure_2d_array(ex_in), self._ensure_2d_array(ex_out)))
                    
                    target_features = self.ma._extract_target_features(clean_examples)
                    fantasy_result, fantasy_ops, fantasy_score = self.ma.fantasy_search(
                        current_grid, target_features, self.si, self.amc, max_depth=2
                    )
                    if fantasy_score > 0.5:
                        new_path = path + fantasy_ops
                        new_g_cost = g_cost + sum(ACTION_COSTS.get(op[0], 1.0) for op in fantasy_ops)
                        next_sim = self.si.robust_similarity_fn(fantasy_result, target_grid)
                        f_n = new_g_cost + (1.0 - next_sim)
                        children.append((f_n, len(new_path), next(counter), (fantasy_result, new_path, new_g_cost)))
                except:
                    pass

            # 3. Expand all primitive ME actions
            for action_name, action_fn in self.me.skills.items():
                action_cost = ACTION_COSTS.get(action_name, 1.0)
                
                try:
                    if action_name == "color_shift":
                        for shift in range(1, min(self.me.num_colors, 5)):  # Limit to avoid explosion
                            params = {"shift": shift, "num_colors": self.me.num_colors}
                            next_grid = self.me.execute(current_grid, action_name, params)
                            new_path = path + [(action_name, params)]
                            new_g_cost = g_cost + action_cost
                            next_sim = self.si.robust_similarity_fn(next_grid, target_grid)
                            f_n = new_g_cost + (1.0 - next_sim)
                            children.append((f_n, len(new_path), next(counter), (next_grid, new_path, new_g_cost)))
                    elif action_name == "object_fill":
                        for fill_color in range(1, min(self.me.num_colors, 5)):
                            params = {"fill_color": fill_color}
                            next_grid = self.me.execute(current_grid, action_name, params)
                            new_path = path + [(action_name, params)]
                            new_g_cost = g_cost + action_cost
                            next_sim = self.si.robust_similarity_fn(next_grid, target_grid)
                            f_n = new_g_cost + (1.0 - next_sim)
                            children.append((f_n, len(new_path), next(counter), (next_grid, new_path, new_g_cost)))
                    else:
                        next_grid = self.me.execute(current_grid, action_name, {})
                        new_path = path + [(action_name, {})]
                        new_g_cost = g_cost + action_cost
                        next_sim = self.si.robust_similarity_fn(next_grid, target_grid)
                        f_n = new_g_cost + (1.0 - next_sim)
                        children.append((f_n, len(new_path), next(counter), (next_grid, new_path, new_g_cost)))
                except Exception as e:
                    self.logger.debug(f"Error applying {action_name}: {e}")
                    continue

            # 4. Expand known general skills from SI
            try:
                for skill in self.si.find_relevant_skills(current_grid, top_k=3):
                    if not skill.get("generalize", False):
                        continue
                    skill_name = skill["skill_name"]
                    action_cost = sum(ACTION_COSTS.get(name, 1.0) for name, _ in skill["action_sequence"]) * 0.9
                    
                    try:
                        next_grid = self.execute_plan(current_grid, [(skill_name, {})])
                        new_path = path + [(skill_name, {})]
                        new_g_cost = g_cost + action_cost
                        next_sim = self.si.robust_similarity_fn(next_grid, target_grid)
                        f_n = new_g_cost + (1.0 - next_sim)
                        children.append((f_n, len(new_path), next(counter), (next_grid, new_path, new_g_cost)))
                    except:
                        continue
            except:
                pass

            # Sort and add best children to queue
            children.sort(key=lambda x: x[0])
            for child in children[:beam_width]:
                if len(child[3][1]) <= max_steps:
                    q.put(child)

        self.logger.info(f"Agent failed to find perfect solution, returning best found: sim={self.si.robust_similarity_fn(best_found[0], target_grid):.2f}")
        return best_found

    @staticmethod
    def pad_to_shape(arr, target_shape, pad_value=0):
        """Pad array to target shape."""
        arr = np.array(arr)
        if arr.ndim != 2:
            # Try to make it 2D
            if arr.size == 0:
                return np.zeros(target_shape, dtype=int)
            side = int(np.sqrt(arr.size))
            if side * side == arr.size:
                arr = arr.reshape((side, side))
            else:
                arr = arr.reshape((-1, 1))
        
        pad_height = target_shape[0] - arr.shape[0]
        pad_width = target_shape[1] - arr.shape[1]
        pad_before = (0, 0)
        pad_after = (max(pad_height, 0), max(pad_width, 0))
        pad_widths = ((0, pad_after[0]), (0, pad_after[1]))
        return np.pad(arr, pad_widths, mode='constant', constant_values=pad_value)[:target_shape[0], :target_shape[1]]

    def act(self, obs, target, examples=None):
        """
        Real-time, shallow search for the best next primitive or abstract action.
        """
        obs = self._ensure_2d_array(obs)
        target = self._ensure_2d_array(target)
        
        result, best_path, valence = self.solve_arc(obs, target, examples=examples, beam_width=1, max_steps=1)
        if best_path:
            first_action = best_path[0]
            self.logger.info(f"Agent act: selected action {first_action}")
            return first_action
        else:
            self.logger.warning("Agent act: No promising action found.")
            return None

    def update(self, state, action, reward, next_state):
        """Update experience and potentially learn new rules."""
        state = self._ensure_2d_array(state)
        next_state = self._ensure_2d_array(next_state)
        
        self.si.store_experience(state, action, reward, next_state)
        
        # If MA is enabled, try to learn patterns from recent experiences
        if self.ma_enabled and len(self.si.experience) % 10 == 0:
            recent_experiences = self.si.experience[-20:]
            examples = []
            for s, a, r, ns in recent_experiences:
                if r > 0.5:
                    examples.append((self._ensure_2d_array(s), self._ensure_2d_array(ns)))
            
            if len(examples) >= 2:
                try:
                    rule = self.ma.learn_transformation_from_examples(examples, self.si)
                    if rule and rule['confidence'] > 0.8:
                        self.logger.info(f"Learned new rule with confidence {rule['confidence']:.2f}")
                except Exception as e:
                    self.logger.debug(f"Failed to learn rule: {e}")

    def dream(self):
        """Enhanced dreaming phase that includes MA's pattern discovery."""
        if getattr(self, "DREAMING_ENABLED", True):
            self.logger.info("--- Dreaming phase: revisiting creative failures and discovering patterns ---")
            
            if self.ma_enabled:
                # MA analyzes past experiences for patterns
                successful_transforms = []
                for state, action, reward, next_state in self.si.experience:
                    if reward > 0.7:
                        try:
                            s = self._ensure_2d_array(state)
                            ns = self._ensure_2d_array(next_state)
                            successful_transforms.append((s, ns))
                        except:
                            continue
                
                if len(successful_transforms) >= 3:
                    # Try to learn general rules from successful transformations
                    try:
                        rule = self.ma.learn_transformation_from_examples(
                            successful_transforms[:10], self.si
                        )
                        if rule and rule['confidence'] > 0.75:
                            self.logger.info(f"Discovered general rule during dreaming: {rule['operations']}")
                    except Exception as e:
                        self.logger.debug(f"Failed to discover rule during dreaming: {e}")
                
                # Original creative dreaming
                try:
                    self.ma.creative_dreaming(self.si, self.amc, max_depth=2)
                except Exception as e:
                    self.logger.debug(f"Creative dreaming error: {e}")
            else:
                self.logger.info("MA disabled, skipping creative dreaming.")
            
            # AMC meta-learning
            try:
                self.amc.meta_learn_weights(self.si)
            except Exception as e:
                self.logger.debug(f"Meta-learning error: {e}")
            
            self.logger.info("--- Dreaming phase complete ---")
        else:
            self.logger.info("--- Dreaming is disabled ---")
