import numpy as np
import logging
from scipy.ndimage import label, find_objects

def identity(grid):
    return np.array(grid)

def flip_horizontal(grid):
    return np.fliplr(np.array(grid))

def flip_vertical(grid):
    return np.flipud(np.array(grid))

def rotate90(grid):
    return np.rot90(np.array(grid))

def color_shift(grid, shift=1, num_colors=10):
    arr = np.array(grid)
    arr[arr > 0] = (arr[arr > 0] + shift) % num_colors
    return arr

@staticmethod
def erase_small_objects(grid, min_size=3):
    arr = np.array(grid)
    if arr.ndim > 2:
        arr = arr[0]  # Use the first slice if 3D
    if arr.ndim != 2:
        arr = arr.reshape((int(np.sqrt(arr.size)), -1)) if arr.size > 0 else np.zeros((1, 1), dtype=int)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x, y] != 0:
                neighbors = 0
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < arr.shape[0] and 0 <= ny < arr.shape[1]:
                        if arr[nx, ny] == arr[x, y]:
                            neighbors += 1
                if neighbors < min_size:
                    arr[x, y] = 0
    return arr

def object_fill(grid, fill_color=1):
    arr = np.array(grid)
    arr[arr > 0] = fill_color
    return arr

def largest_object_crop(grid):
    arr = np.array(grid)
    nonzero = np.argwhere(arr)
    if nonzero.size == 0:
        return arr
    (x0, y0), (x1, y1) = nonzero.min(0), nonzero.max(0) + 1
    return arr[x0:x1, y0:y1]

def mirror_diagonal(grid):
    return np.array(grid).T

def extract_pattern(grid, pattern_size=3):
    arr = np.array(grid)
    # If arr is a list of arrays, use the first one
    if isinstance(arr, list) or arr.ndim > 2:
        arr = np.array(arr[0]) if len(arr) > 0 else np.zeros((pattern_size, pattern_size), dtype=int)
    if arr.ndim != 2:
        arr = arr.reshape((arr.size, 1)) if arr.size > 0 else np.zeros((pattern_size, pattern_size), dtype=int)
    h, w = arr.shape
    if h < pattern_size or w < pattern_size:
        return arr
    patterns = {}
    for i in range(h - pattern_size + 1):
        for j in range(w - pattern_size + 1):
            pattern = arr[i:i+pattern_size, j:j+pattern_size]
            pattern_key = tuple(pattern.flatten())
            patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
    if patterns:
        most_common = max(patterns.items(), key=lambda x: x[1])[0]
        return np.array(most_common).reshape(pattern_size, pattern_size)
    return arr[:pattern_size, :pattern_size]

def repeat_pattern(grid, pattern_size=3):
    pattern = extract_pattern(grid, pattern_size)
    h, w = grid.shape
    tiled = np.tile(pattern, (h // pattern_size + 1, w // pattern_size + 1))
    return tiled[:h, :w]

def color_map(grid, mapping):
    arr = np.array(grid)
    new_grid = np.zeros_like(arr)
    for old_color, new_color in mapping.items():
        new_grid[arr == old_color] = new_color
    return new_grid

def flood_fill(grid, start_pos=(0, 0), new_color=1):
    arr = np.array(grid)
    if not (0 <= start_pos[0] < arr.shape[0] and 0 <= start_pos[1] < arr.shape[1]):
        return arr
    original_color = arr[start_pos]
    if isinstance(original_color, np.ndarray):
        original_color = original_color.item()  # Convert to scalar if needed
    if original_color == new_color:
        return arr
    stack = [start_pos]
    while stack:
        x, y = stack.pop()
        if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1] and arr[x, y] == original_color:
            arr[x, y] = new_color
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return arr

def extract_objects(grid, min_size=2):
    arr = np.array(grid)
    labeled, num_objects = label(arr > 0)
    objects = []
    for i in range(1, num_objects + 1):
        obj_mask = (labeled == i)
        if np.sum(obj_mask) >= min_size:
            slices = find_objects(obj_mask)[0]
            obj_grid = arr[slices] * obj_mask[slices]
            objects.append(obj_grid)
    return objects if objects else [arr]

def compose_objects(grid, objects):
    arr = np.zeros_like(grid)
    for obj in objects:
        h, w = obj.shape
        arr[:h, :w] += obj
    return arr

def symmetrize(grid):
    arr = np.array(grid)
    return (arr + np.fliplr(arr) + np.flipud(arr)) // 3

def extract_border(grid):
    arr = np.array(grid)
    border = np.zeros_like(arr)
    border[0, :] = arr[0, :]
    border[-1, :] = arr[-1, :]
    border[:, 0] = arr[:, 0]
    border[:, -1] = arr[:, -1]
    return border

def fill_border(grid, color=1):
    arr = np.array(grid)
    arr[0, :] = color
    arr[-1, :] = color
    arr[:, 0] = color
    arr[:, -1] = color
    return arr

class MAEngine:
    """
    Model Actor (MA) Engine for UPCA.
    Abstract/Fantasy Engine: learns, composes, and applies abstract transformation rules.
    """

    def __init__(self, me_engine):
        self.me_engine = me_engine
        self.logger = logging.getLogger("MAEngine")
        self.learned_rules = []
        self.abstracted_patterns = {}
        self.parameterized = {"color_shift", "object_fill", "flood_fill"}
        self.transformations = [
            identity,
            flip_horizontal,
            flip_vertical,
            rotate90,
            color_shift,
            erase_small_objects,
            self.me_engine.crop_nonzero,
            object_fill,
            largest_object_crop,
            mirror_diagonal,
            extract_pattern,
            repeat_pattern,
            color_map,
            flood_fill,
            extract_objects,
            compose_objects,
            symmetrize,
            extract_border,
            fill_border,
        ]

    def get_transformation_by_name(self, name):
        for fn in self.transformations:
            if hasattr(fn, '__name__') and fn.__name__ == name:
                return fn
        return None
    
    def propose_action(self, state, si=None, amc=None, latent=False):
        """
        Propose the best single-step action (with parameters) for the given state.
        If latent=True, state is a latent vector; otherwise, it's a grid.
        """
        best_score = -float('inf')
        best_action = None
        best_state = None

        for action_fn in self.transformations:
            action_name = action_fn.__name__ if hasattr(action_fn, '__name__') else str(action_fn)
            
            if action_name == "color_shift":
                for shift in range(1, self.me_engine.num_colors):
                    if latent:
                        continue
                    else:
                        sim_state = action_fn(state, shift=shift, num_colors=self.me_engine.num_colors)
                    score = amc.evaluate_valence(sim_state) if amc else np.count_nonzero(sim_state)
                    self.logger.debug(f"MA propose: {action_name} (shift={shift}) score={score:.2f}")
                    if score > best_score:
                        best_score = score
                        best_action = (action_fn, {"shift": shift, "num_colors": self.me_engine.num_colors})
                        best_state = sim_state
            elif action_name == "object_fill":
                for fill_color in range(1, self.me_engine.num_colors):
                    if latent:
                        continue
                    else:
                        sim_state = action_fn(state, fill_color=fill_color)
                    score = amc.evaluate_valence(sim_state) if amc else np.count_nonzero(sim_state)
                    self.logger.debug(f"MA propose: {action_name} (fill_color={fill_color}) score={score:.2f}")
                    if score > best_score:
                        best_score = score
                        best_action = (action_fn, {"fill_color": fill_color})
                        best_state = sim_state
            else:
                if latent:
                    continue
                else:
                    sim_state = self.me_engine.apply_action(state, action_fn)
                score = amc.evaluate_valence(sim_state) if amc else np.count_nonzero(sim_state)
                self.logger.debug(f"MA propose: {action_name} score={score:.2f}")
                if score > best_score:
                    best_score = score
                    best_action = (action_fn, {})
                    best_state = sim_state

        if best_action is None:
            self.logger.warning("MA: No action proposed (all scores -inf).")
        return best_action, best_state, best_score

    def plan_sequence(self, state, max_depth=2, si=None, amc=None, log_attempts=None, latent=False):
        """
        Propose the best multi-step action sequence for the given state.
        """
        def recursive_plan(current_state, depth, action_seq):
            if depth == 0:
                score = amc.evaluate_valence(current_state) if amc else np.count_nonzero(current_state)
                if log_attempts is not None:
                    log_attempts.append((action_seq, current_state, score))
                if si is not None:
                    si.store_experience(state, action_seq, score, current_state)
                self.logger.debug(f"MA plan_sequence: {action_seq} score={score:.2f}")
                return action_seq, current_state, score
            
            best = (action_seq, current_state, -float('inf'))
            for action_fn in self.transformations:
                action_name = action_fn.__name__ if hasattr(action_fn, '__name__') else str(action_fn)
                
                if action_name == "color_shift":
                    for shift in range(1, self.me_engine.num_colors):
                        if latent:
                            continue
                        else:
                            next_state = action_fn(current_state, shift=shift, num_colors=self.me_engine.num_colors)
                        seq, state_out, score = recursive_plan(next_state, depth-1, action_seq + [(action_name, {"shift": shift, "num_colors": self.me_engine.num_colors})])
                        if score > best[2]:
                            best = (seq, state_out, score)
                elif action_name == "object_fill":
                    for fill_color in range(1, self.me_engine.num_colors):
                        if latent:
                            continue
                        else:
                            next_state = action_fn(current_state, fill_color=fill_color)
                        seq, state_out, score = recursive_plan(next_state, depth-1, action_seq + [(action_name, {"fill_color": fill_color})])
                        if score > best[2]:
                            best = (seq, state_out, score)
                else:
                    if latent:
                        continue
                    else:
                        next_state = self.me_engine.apply_action(current_state, action_fn)
                    seq, state_out, score = recursive_plan(next_state, depth-1, action_seq + [(action_name, {})])
                    if score > best[2]:
                        best = (seq, state_out, score)
            return best

        log_attempts = log_attempts if log_attempts is not None else []
        return recursive_plan(state, max_depth, [])

    def creative_dreaming(self, si, amc, max_depth=2):
        """Creative dreaming phase for discovering new patterns."""
        print("\n--- Dreaming phase: revisiting creative failures and merging rules ---")
        for (state, action, reward, next_state) in si.experience[-20:]:  # Only recent experiences
            for fn1, fn2 in [(f1, f2) for f1 in self.transformations[:5] for f2 in self.transformations[:5]]:
                fn1_name = fn1.__name__ if hasattr(fn1, '__name__') else str(fn1)
                fn2_name = fn2.__name__ if hasattr(fn2, '__name__') else str(fn2)
                
                try:
                    if fn1_name == "color_shift":
                        for shift1 in range(1, min(3, self.me_engine.num_colors)):
                            candidate = fn1(state, shift=shift1, num_colors=self.me_engine.num_colors)
                            candidate = fn2(candidate) if fn2_name not in self.parameterized else candidate
                            if reward is None:
                                reward = 0.0
                            score = amc.evaluate_valence(candidate) if amc else np.count_nonzero(candidate)
                            if score > reward:
                                si.store_experience(state, [(fn1_name, {"shift": shift1}), (fn2_name, {})], score, candidate)
                                print(f"Dreamed: [{fn1_name}(shift={shift1}), {fn2_name}], score: {score:.2f}")
                    else:
                        candidate = fn2(fn1(state))
                        if reward is None:
                            reward = 0.0
                        score = amc.evaluate_valence(candidate) if amc else np.count_nonzero(candidate)
                        if score > reward:
                            si.store_experience(state, [(fn1_name, {}), (fn2_name, {})], score, candidate)
                            print(f"Dreamed: [{fn1_name}, {fn2_name}], score: {score:.2f}")
                except Exception as e:
                    self.logger.debug(f"Dream error: {e}")
                    continue
        
    def learn_transformation_from_examples(self, examples, si):
        """Learn the transformation rule from input-output examples."""
        if not examples:
            return None
            
        # Analyze what changes between input and output
        transformations = []
        for input_grid, output_grid in examples:
            trans = self._analyze_transformation(input_grid, output_grid)
            transformations.append(trans)
        
        # Find common pattern across examples
        rule = self._abstract_common_rule(transformations)
        
        if rule:
            self.learned_rules.append(rule)
            # Store in scaffold
            si.store_transformation(
                examples[0][0], 
                rule['operations'],
                examples[0][1],
                generalize=True
            )
            
        return rule
    
    def _analyze_transformation(self, input_grid, output_grid):
        """Analyze what transformation occurred between input and output."""
        analysis = {
            'size_change': self._analyze_size_change(input_grid, output_grid),
            'color_mapping': self._analyze_color_mapping(input_grid, output_grid),
            'spatial_transform': self._analyze_spatial_transform(input_grid, output_grid),
            'object_changes': self._analyze_object_changes(input_grid, output_grid),
            'pattern_completion': self._analyze_pattern_completion(input_grid, output_grid),
            'symmetry': self._analyze_symmetry_operations(input_grid, output_grid)
        }
        return analysis
    
    def _analyze_symmetry_operations(self, input_grid, output_grid):
        """Analyze symmetry-related transformations."""
        # Check if output is a symmetrized version of input
        if np.array_equal(output_grid, np.fliplr(input_grid)):
            return {'type': 'horizontal_mirror'}
        if np.array_equal(output_grid, np.flipud(input_grid)):
            return {'type': 'vertical_mirror'}
        if np.array_equal(output_grid, input_grid.T):
            return {'type': 'diagonal_mirror'}
        
        # Check if output completes symmetry
        if self._is_symmetric(output_grid) and not self._is_symmetric(input_grid):
            if self._is_partial_symmetry(input_grid):
                return {'type': 'symmetry_completion'}
        
        # Check if output is symmetrized (averaged) version
        symmetrized = (input_grid + np.fliplr(input_grid) + np.flipud(input_grid)) // 3
        if np.array_equal(output_grid, symmetrized):
            return {'type': 'symmetrize_average'}
        
        return {'type': 'none'}
    
    def _analyze_size_change(self, input_grid, output_grid):
        """Detect if output size is related to input properties."""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        # Check for scaling
        if out_h == in_h and out_w == in_w:
            return {'type': 'same_size'}
        elif out_h == in_h * 2 and out_w == in_w * 2:
            return {'type': 'double_size'}
        elif out_h * 2 == in_h and out_w * 2 == in_w:
            return {'type': 'half_size'}
        
        # Check if size relates to object count
        input_objects = self._extract_objects(input_grid)
        if len(input_objects) > 0:
            if out_h == len(input_objects) or out_w == len(input_objects):
                return {'type': 'size_from_object_count', 'count': len(input_objects)}
        
        return {'type': 'custom', 'input_size': (in_h, in_w), 'output_size': (out_h, out_w)}
    
    def _analyze_color_mapping(self, input_grid, output_grid):
        """Detect color transformation rules."""
        input_colors = set(np.unique(input_grid)) - {0}
        output_colors = set(np.unique(output_grid)) - {0}
        
        # Direct mapping
        if len(input_colors) == len(output_colors) and len(input_colors) > 0:
            color_map = {}
            for in_color in input_colors:
                # Find where this color appears in input
                in_mask = (input_grid == in_color)
                # See what colors appear in same positions in output
                out_colors_at_pos = output_grid[in_mask]
                unique_out = np.unique(out_colors_at_pos)
                if len(unique_out) == 1:
                    color_map[int(in_color)] = int(unique_out[0])
            
            if len(color_map) == len(input_colors):
                return {'type': 'direct_mapping', 'map': color_map}
        
        # Color reduction
        if len(output_colors) < len(input_colors):
            return {'type': 'color_reduction', 'from': input_colors, 'to': output_colors}
        
        return {'type': 'complex'}
    
    def _analyze_spatial_transform(self, input_grid, output_grid):
        """Detect spatial transformations like rotations, flips, shifts."""
        if input_grid.shape != output_grid.shape:
            return {'type': 'none'}
        
        # Check rotations
        for k in range(4):
            if np.array_equal(np.rot90(input_grid, k), output_grid):
                return {'type': 'rotation', 'k': k}
        
        # Check flips
        if np.array_equal(np.fliplr(input_grid), output_grid):
            return {'type': 'flip', 'axis': 'horizontal'}
        if np.array_equal(np.flipud(input_grid), output_grid):
            return {'type': 'flip', 'axis': 'vertical'}
        
        # Check if it's a subset/superset
        if self._is_subset(input_grid, output_grid):
            return {'type': 'subset_copy'}
        
        return {'type': 'none'}
    
    def _analyze_object_changes(self, input_grid, output_grid):
        """Analyze how individual objects change."""
        input_objects = self._extract_objects(input_grid)
        output_objects = self._extract_objects(output_grid)
        
        changes = {
            'count_change': len(output_objects) - len(input_objects),
            'objects_modified': [],
            'objects_removed': [],
            'objects_added': []
        }
        
        # Match objects between input and output
        if input_objects and output_objects:
            for in_obj in input_objects:
                matched = False
                for out_obj in output_objects:
                    if self._objects_match(in_obj, out_obj):
                        matched = True
                        if not np.array_equal(in_obj['mask'], out_obj['mask']):
                            changes['objects_modified'].append({
                                'from': in_obj,
                                'to': out_obj,
                                'change_type': self._classify_object_change(in_obj, out_obj)
                            })
                        break
                
                if not matched:
                    changes['objects_removed'].append(in_obj)
        
        return changes
    
    def _analyze_pattern_completion(self, input_grid, output_grid):
        """Check if output completes a pattern started in input."""
        # Check for symmetry completion
        if self._is_partial_symmetry(input_grid) and self._is_symmetric(output_grid):
            return {'type': 'symmetry_completion'}
        
        # Check for pattern repetition
        if self._contains_pattern(output_grid, input_grid):
            return {'type': 'pattern_repetition'}
        
        return {'type': 'none'}
    
    def _abstract_common_rule(self, transformations):
        """Find the common rule across multiple transformation analyses."""
        if not transformations:
            return None
        
        # Check if all have same size change
        size_changes = [t['size_change']['type'] for t in transformations]
        if all(s == size_changes[0] for s in size_changes):
            size_rule = size_changes[0]
        else:
            size_rule = 'variable'
        
        # Check if all have same color mapping
        color_mappings = [t['color_mapping']['type'] for t in transformations]
        if all(c == color_mappings[0] for c in color_mappings):
            color_rule = transformations[0]['color_mapping']
        else:
            color_rule = {'type': 'variable'}
        
        # Build abstract rule
        rule = {
            'size_rule': size_rule,
            'color_rule': color_rule,
            'operations': self._generate_operations(transformations),
            'confidence': self._compute_rule_confidence(transformations)
        }
        
        return rule
    
    def _generate_operations(self, transformations):
        """Generate executable operations from transformation analysis."""
        operations = []
        
        # Prioritize consistent transformations
        if all(t['spatial_transform']['type'] != 'none' for t in transformations):
            spatial = transformations[0]['spatial_transform']
            if spatial['type'] == 'rotation':
                operations.append(('rotate', {'k': spatial['k']}))
            elif spatial['type'] == 'flip':
                operations.append(('flip', {'axis': spatial['axis']}))
        
        # Add symmetry operations
        symmetry_ops = [t['symmetry']['type'] for t in transformations]
        if all(s != 'none' for s in symmetry_ops) and all(s == symmetry_ops[0] for s in symmetry_ops):
            sym_type = symmetry_ops[0]
            if sym_type == 'horizontal_mirror':
                operations.append(('flip_horizontal', {}))
            elif sym_type == 'vertical_mirror':
                operations.append(('flip_vertical', {}))
            elif sym_type == 'diagonal_mirror':
                operations.append(('mirror_diagonal', {}))
        
        # Add object-based operations
        obj_changes = [t['object_changes'] for t in transformations]
        if all(o['count_change'] == obj_changes[0]['count_change'] for o in obj_changes):
            if obj_changes[0]['count_change'] < 0:
                operations.append(('remove_objects', {'count': -obj_changes[0]['count_change']}))
        
        # Add color operations
        color_rules = [t['color_mapping'] for t in transformations]
        if all(c['type'] == 'direct_mapping' for c in color_rules):
            operations.append(('color_map', {'mapping': color_rules[0]['map']}))
        
        return operations
    
    def apply_learned_rule(self, input_grid, rule):
        """Apply a learned rule to a new input."""
        current_grid = input_grid.copy()
        
        for op_name, params in rule['operations']:
            if op_name == 'rotate':
                current_grid = np.rot90(current_grid, params.get('k', 1))
            elif op_name == 'flip':
                if params.get('axis') == 'horizontal':
                    current_grid = np.fliplr(current_grid)
                else:
                    current_grid = np.flipud(current_grid)
            elif op_name == 'flip_horizontal':
                current_grid = np.fliplr(current_grid)
            elif op_name == 'flip_vertical':
                current_grid = np.flipud(current_grid)
            elif op_name == 'mirror_diagonal':
                current_grid = current_grid.T
            elif op_name == 'color_map':
                new_grid = np.zeros_like(current_grid)
                for old_color, new_color in params.get('mapping', {}).items():
                    new_grid[current_grid == int(old_color)] = int(new_color)
                current_grid = new_grid
            elif op_name == 'remove_objects':
                # Remove smallest objects
                objects = self._extract_objects(current_grid)
                objects.sort(key=lambda x: x['size'])
                for i in range(min(params.get('count', 0), len(objects))):
                    current_grid[objects[i]['mask']] = 0
        
        return current_grid
    
    def fantasy_search(self, input_grid, target_features, si, amc, max_depth=3):
        """
        Search for transformation sequences that achieve target features.
        This is the counterfactual reasoning component.
        """
        # Use beam search to find transformations
        beam_width = 10
        beam = [(input_grid, [], 0.0)]  # (grid, operations, score)
        
        for depth in range(max_depth):
            new_beam = []
            
            for current_grid, ops, score in beam:
                # Try applying learned rules
                for rule in self.learned_rules:
                    try:
                        new_grid = self.apply_learned_rule(current_grid, rule)
                        new_ops = ops + [('learned_rule', rule)]
                        new_score = self._score_against_target(new_grid, target_features, amc)
                        new_beam.append((new_grid, new_ops, new_score))
                    except:
                        continue
                
                # Try basic transformations
                for transform_fn in self.transformations[:10]:  # Limit to avoid explosion
                    try:
                        transform_name = transform_fn.__name__ if hasattr(transform_fn, '__name__') else str(transform_fn)
                        if transform_name in self.parameterized:
                            continue  # Skip parameterized for now
                        new_grid = transform_fn(current_grid)
                        new_ops = ops + [(transform_name, {})]
                        new_score = self._score_against_target(new_grid, target_features, amc)
                        new_beam.append((new_grid, new_ops, new_score))
                    except:
                        continue
            
            # Keep top beam_width candidates
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_width]
            
            # Early stopping if we found a good solution
            if beam and beam[0][2] > 0.95:
                break
        
        return beam[0] if beam else (input_grid, [], 0.0)
    
    def _extract_objects(self, grid):
        """Extract individual objects from grid."""
        try:
            labeled, num_objects = label(grid > 0)
            objects = []
            
            for i in range(1, num_objects + 1):
                mask = (labeled == i)
                if np.sum(mask) == 0:
                    continue
                color_values = grid[mask]
                if len(color_values) == 0:
                    continue
                color = int(color_values[0])
                bbox = find_objects(mask)[0]
                
                objects.append({
                    'mask': mask,
                    'color': color,
                    'bbox': bbox,
                    'size': np.sum(mask),
                    'center': (
                        (bbox[0].start + bbox[0].stop) / 2,
                        (bbox[1].start + bbox[1].stop) / 2
                    )
                })
            
            return objects
        except:
            return []
    
    def _is_subset(self, small, large):
        """Check if small grid appears in large grid."""
        if small.shape[0] > large.shape[0] or small.shape[1] > large.shape[1]:
            return False
        
        for i in range(large.shape[0] - small.shape[0] + 1):
            for j in range(large.shape[1] - small.shape[1] + 1):
                if np.array_equal(large[i:i+small.shape[0], j:j+small.shape[1]], small):
                    return True
        return False
    
    def _objects_match(self, obj1, obj2):
        """Check if two objects are the same (possibly moved)."""
        return (obj1['color'] == obj2['color'] and 
                abs(obj1['size'] - obj2['size']) < 3)
    
    def _classify_object_change(self, obj1, obj2):
        """Classify how an object changed."""
        if obj1['center'] != obj2['center']:
            return 'moved'
        elif obj1['size'] != obj2['size']:
            return 'resized'
        else:
            return 'modified'
    
    def _is_symmetric(self, grid):
        """Check if grid has symmetry."""
        return (np.array_equal(grid, np.fliplr(grid)) or 
                np.array_equal(grid, np.flipud(grid)) or
                np.array_equal(grid, grid.T))
    
    def _is_partial_symmetry(self, grid):
        """Check if grid has partial symmetry that could be completed."""
        h, w = grid.shape
        
        # Check horizontal partial symmetry
        if w % 2 == 0:
            left_half = grid[:, :w//2]
            right_half = grid[:, w//2:]
            if np.sum(left_half > 0) > 0 and np.sum(right_half > 0) == 0:
                return True
        
        # Check vertical partial symmetry
        if h % 2 == 0:
            top_half = grid[:h//2, :]
            bottom_half = grid[h//2:, :]
            if np.sum(top_half > 0) > 0 and np.sum(bottom_half > 0) == 0:
                return True
        
        return False
    
    def _contains_pattern(self, large, small):
        """Check if large grid contains small grid as a pattern."""
        return self._is_subset(small, large)
    
    def _compute_rule_confidence(self, transformations):
        """Compute confidence in the abstracted rule."""
        if len(transformations) < 2:
            return 0.5
        
        # Check consistency across transformations
        consistency_scores = []
        
        # Size consistency
        size_types = [t['size_change']['type'] for t in transformations]
        if len(size_types) > 0:
            size_consistency = 1.0 - (len(set(size_types)) - 1) / len(size_types)
            consistency_scores.append(size_consistency)
        
        # Color consistency
        color_types = [t['color_mapping']['type'] for t in transformations]
        if len(color_types) > 0:
            color_consistency = 1.0 - (len(set(color_types)) - 1) / len(color_types)
            consistency_scores.append(color_consistency)
        
        # Symmetry consistency
        symmetry_types = [t['symmetry']['type'] for t in transformations]
        if len(symmetry_types) > 0:
            symmetry_consistency = 1.0 - (len(set(symmetry_types)) - 1) / len(symmetry_types)
            consistency_scores.append(symmetry_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _score_against_target(self, grid, target_features, amc):
        """Score how well a grid matches target features."""
        score = 0.0
        
        if 'size' in target_features and target_features['size'] is not None:
            if grid.shape == target_features['size']:
                score += 0.3
        
        if 'colors' in target_features:
            grid_colors = set(np.unique(grid)) - {0}
            target_colors = set(target_features['colors'])
            if len(target_colors) > 0:
                color_overlap = len(grid_colors & target_colors) / len(target_colors)
                score += 0.3 * color_overlap
        
        if 'object_count' in target_features:
            objects = self._extract_objects(grid)
            if len(objects) == target_features['object_count']:
                score += 0.2
        
        # Add AMC valence
        if amc:
            try:
                valence = amc.evaluate_valence(grid)
                score += 0.2 * max(0, valence)
            except:
                pass
        
        return score
    
    def propose_creative_solutions(self, input_grid, examples, si, amc):
        """
        Main entry point for MA engine to propose solutions.
        Learns from examples and applies to new input.
        """
        # Learn transformation rule from examples
        rule = self.learn_transformation_from_examples(examples, si)
        
        if rule and rule['confidence'] > 0.7:
            # Apply learned rule
            try:
                output = self.apply_learned_rule(input_grid, rule)
                self.logger.info(f"MA: Applied learned rule with confidence {rule['confidence']:.2f}")
                return output, rule['operations']
            except Exception as e:
                self.logger.warning(f"MA: Failed to apply learned rule: {e}")
        
        # Fall back to fantasy search
        self.logger.info("MA: Low confidence in learned rule, using fantasy search")
        
        # Extract target features from examples
        target_features = self._extract_target_features(examples)
        
        # Search for transformation
        output, operations, score = self.fantasy_search(
            input_grid, target_features, si, amc
        )
        
        return output, operations
    
    def _extract_target_features(self, examples):
        """Extract common features from output examples."""
        if not examples:
            return {}
        
        output_grids = [output for _, output in examples]
        
        features = {
            'size': output_grids[0].shape if all(g.shape == output_grids[0].shape for g in output_grids) else None,
            'colors': list(set().union(*[set(np.unique(g)) - {0} for g in output_grids])),
            'object_count': len(self._extract_objects(output_grids[0]))
        }
        
        return features
