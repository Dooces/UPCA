import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.ndimage import label, find_objects

class MEEngine(nn.Module):
    """
    Model Executor (ME) Engine for UPCA.
    - Encodes/decodes grid states (dynamic grid size, convolutional VAE).
    - Executes single, specified, parameterized, and object-centric skills.
    - Logs every execution and result.
    - Ready for latent-space integration.
    """

    def __init__(self, latent_dim=16, num_colors=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.logger = logging.getLogger("MEEngine")

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1, padding=1),
        )

        # Register all available skills
        self.skills = {
            "identity": self.identity,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "rotate90": self.rotate90,
            "color_shift": self.color_shift,
            "erase_small_objects": self.erase_small_objects,
            "crop_nonzero": self.crop_nonzero,
            "object_fill": self.object_fill,
            "largest_object_crop": self.largest_object_crop,
            "mirror_diagonal": self.mirror_diagonal,
            "extract_pattern": self.extract_pattern,
            "repeat_pattern": self.repeat_pattern,
            "color_map": self.color_map,
            "flood_fill": self.flood_fill,
            "extract_objects": self.extract_objects,
            "compose_objects": self.compose_objects,
            "symmetrize": self.symmetrize,
            "extract_border": self.extract_border,
            "fill_border": self.fill_border,
        }

    # --- Perceptual Inference (VAE) ---
    def encode(self, grid):
        if isinstance(grid, np.ndarray):
            grid = grid.copy()
            grid = torch.tensor(grid, dtype=torch.float32)
        if grid.ndim == 2:
            grid = grid.unsqueeze(0)
        if grid.ndim == 3:
            grid = grid.unsqueeze(1)
        h = self.encoder(grid)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, out_shape):
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        h = self.fc_decode(z).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(h)
        out = F.interpolate(out, size=out_shape, mode='nearest')
        out = torch.round(torch.clamp(out, 0, self.num_colors - 1))
        return out.squeeze(1).int()

    def forward(self, grid):
        z = self.encode(grid)
        out_shape = grid.shape[-2:]
        recon = self.decode(z, out_shape)
        return recon, z

    def loss(self, grid, recon, mu, logvar):
        recon_loss = F.mse_loss(recon.float(), grid.float(), reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.0001 * kld

    # --- Motor Inference/Execution: Single-Step Skill Application ---
    def execute(self, grid, skill_name, params=None):
        """Execute a skill with proper error handling."""
        fn = self.skills.get(skill_name)
        if fn is None:
            self.logger.warning(f"ME: Unknown skill '{skill_name}'. Returning input unchanged.")
            return grid
        
        try:
            if params is None:
                result = fn(grid)
                self.logger.debug(f"ME: Executed {skill_name} (no params).")
            else:
                result = fn(grid, **params)
                self.logger.debug(f"ME: Executed {skill_name} with params {params}.")
            return result
        except Exception as e:
            self.logger.error(f"ME: Error executing {skill_name}: {e}")
            return grid

    # --- Static Skill Functions ---
    @staticmethod
    def identity(grid):
        return np.array(grid)

    @staticmethod
    def flip_horizontal(grid):
        return np.fliplr(np.array(grid))

    @staticmethod
    def flip_vertical(grid):
        return np.flipud(np.array(grid))

    @staticmethod
    def rotate90(grid):
        return np.rot90(np.array(grid))

    @staticmethod
    def color_shift(grid, shift=1, num_colors=10):
        arr = np.array(grid)
        if np.any(arr > 0):
            arr[arr > 0] = ((arr[arr > 0] - 1 + shift) % num_colors) + 1
        return arr

    @staticmethod
    def erase_small_objects(grid, min_size=3):
        arr = np.array(grid)
        labeled, num_objects = label(arr > 0)
        for i in range(1, num_objects + 1):
            if np.sum(labeled == i) < min_size:
                arr[labeled == i] = 0
        return arr

    @staticmethod
    def crop_nonzero(grid):
        arr = np.array(grid)
        if arr.ndim != 2 or np.all(arr == 0):
            return arr
        nonzero = np.argwhere(arr > 0)
        if nonzero.size == 0:
            return arr
        (x0, y0), (x1, y1) = nonzero.min(0), nonzero.max(0) + 1
        return arr[x0:x1, y0:y1]

    @staticmethod
    def object_fill(grid, fill_color=1):
        arr = np.array(grid)
        arr[arr > 0] = fill_color
        return arr

    @staticmethod
    def largest_object_crop(grid):
        arr = np.array(grid)
        labeled, num_objects = label(arr > 0)
        if num_objects == 0:
            return arr
        
        # Find largest object
        max_size = 0
        max_label = 0
        for i in range(1, num_objects + 1):
            size = np.sum(labeled == i)
            if size > max_size:
                max_size = size
                max_label = i
        
        # Crop to largest object
        if max_label > 0:
            obj_mask = (labeled == max_label)
            nonzero = np.argwhere(obj_mask)
            if nonzero.size > 0:
                (x0, y0), (x1, y1) = nonzero.min(0), nonzero.max(0) + 1
                return arr[x0:x1, y0:y1] * obj_mask[x0:x1, y0:y1]
        
        return arr

    @staticmethod
    def mirror_diagonal(grid):
        """Mirror along main diagonal"""
        return np.array(grid).T

    @staticmethod
    def extract_pattern(grid, pattern_size=3):
        """Extract most common pattern of given size"""
        arr = np.array(grid)
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

    @staticmethod
    def repeat_pattern(grid, pattern_size=3):
        """Repeat the most common pattern across the grid"""
        pattern = MEEngine.extract_pattern(grid, pattern_size)
        h, w = grid.shape
        tiled = np.tile(pattern, (h // pattern_size + 1, w // pattern_size + 1))
        return tiled[:h, :w]

    @staticmethod
    def color_map(grid, mapping=None):
        """Map colors according to provided mapping"""
        arr = np.array(grid)
        if mapping is None:
            return arr
        
        new_grid = arr.copy()
        for old_color, new_color in mapping.items():
            new_grid[arr == int(old_color)] = int(new_color)
        return new_grid

    @staticmethod
    def flood_fill(grid, start_pos=(0, 0), new_color=1):
        """Flood fill from start position"""
        arr = np.array(grid).copy()
        h, w = arr.shape
        
        if not (0 <= start_pos[0] < h and 0 <= start_pos[1] < w):
            return arr
        
        original_color = arr[start_pos[0], start_pos[1]]
        if original_color == new_color:
            return arr
        
        stack = [start_pos]
        while stack:
            x, y = stack.pop()
            if 0 <= x < h and 0 <= y < w and arr[x, y] == original_color:
                arr[x, y] = new_color
                stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return arr

    @staticmethod
    def extract_objects(grid, min_size=2):
        """Extract individual objects as separate grids"""
        arr = np.array(grid)
        labeled, num_objects = label(arr > 0)
        objects = []
        
        for i in range(1, num_objects + 1):
            obj_mask = (labeled == i)
            if np.sum(obj_mask) >= min_size:
                slices = find_objects(labeled == i)
                if slices:
                    slc = slices[0]
                    obj_grid = arr[slc] * obj_mask[slc]
                    objects.append(obj_grid)
        
        return objects if objects else [arr]

    @staticmethod
    def compose_objects(grid, objects=None):
        """Compose objects back into a grid"""
        if objects is None:
            return grid
        
        arr = np.zeros_like(grid)
        for obj in objects:
            if isinstance(obj, np.ndarray):
                h, w = min(obj.shape[0], arr.shape[0]), min(obj.shape[1], arr.shape[1])
                arr[:h, :w] = np.maximum(arr[:h, :w], obj[:h, :w])
        return arr

    @staticmethod
    def symmetrize(grid):
        """Make grid symmetric"""
        arr = np.array(grid)
        # Average with horizontal and vertical flips
        h_flip = np.fliplr(arr)
        v_flip = np.flipud(arr)
        return np.maximum.reduce([arr, h_flip, v_flip])

    @staticmethod
    def extract_border(grid):
        """Extract only the border pixels"""
        arr = np.array(grid)
        border = np.zeros_like(arr)
        if arr.shape[0] > 0 and arr.shape[1] > 0:
            border[0, :] = arr[0, :]
            border[-1, :] = arr[-1, :]
            border[:, 0] = arr[:, 0]
            border[:, -1] = arr[:, -1]
        return border

    @staticmethod
    def fill_border(grid, color=1):
        """Fill the border with specified color"""
        arr = np.array(grid).copy()
        if arr.shape[0] > 0 and arr.shape[1] > 0:
            arr[0, :] = color
            arr[-1, :] = color
            arr[:, 0] = color
            arr[:, -1] = color
        return arr

    def apply_action(self, grid, action_fn, params=None):
        """Apply an action function with parameters"""
        if callable(action_fn):
            if params is None:
                return action_fn(grid)
            else:
                # Only pass parameters that the function accepts
                import inspect
                sig = inspect.signature(action_fn)
                valid_params = {k: v for k, v in params.items() if k in sig.parameters}
                return action_fn(grid, **valid_params)
        else:
            self.logger.warning(f"Action {action_fn} is not callable")
            return grid

    def recall(self, test_input, si, amc, log=True):
        """Recall and apply transformations from scaffold"""
        best_valence = -float('inf')
        best_grid = None
        best_action_seq = None
        
        for trans in si.transformations:
            try:
                candidate_grid = np.array(test_input)
                action_sequence = []
                
                for name, params in trans["action_sequence"]:
                    fn = self.skills.get(name)
                    if fn is None:
                        continue
                    
                    if name == "color_shift":
                        best_val = -float('inf')
                        best_candidate = None
                        best_params = None
                        for shift in range(1, self.num_colors):
                            try:
                                candidate = fn(candidate_grid, shift=shift, num_colors=self.num_colors)
                                valence = amc.evaluate_valence(candidate, 0.0, 0.0, si=si, skill_name=name, param=shift)
                                if log:
                                    self.logger.debug(f"ME recall: tried {name} (shift={shift}) valence={valence:.2f}")
                                if valence > best_val:
                                    best_val = valence
                                    best_candidate = candidate
                                    best_params = {"shift": shift, "num_colors": self.num_colors}
                            except Exception as e:
                                self.logger.debug(f"Error in color_shift: {e}")
                                continue
                        
                        if best_candidate is not None:
                            candidate_grid = best_candidate
                            action_sequence.append((name, best_params))
                    
                    elif name == "object_fill":
                        best_val = -float('inf')
                        best_candidate = None
                        best_params = None
                        for fill_color in range(1, self.num_colors):
                            try:
                                candidate = fn(candidate_grid, fill_color=fill_color)
                                valence = amc.evaluate_valence(candidate, 0.0, 0.0, si=si, skill_name=name, param=fill_color)
                                if log:
                                    self.logger.debug(f"ME recall: tried {name} (fill_color={fill_color}) valence={valence:.2f}")
                                if valence > best_val:
                                    best_val = valence
                                    best_candidate = candidate
                                    best_params = {"fill_color": fill_color}
                            except Exception as e:
                                self.logger.debug(f"Error in object_fill: {e}")
                                continue
                        
                        if best_candidate is not None:
                            candidate_grid = best_candidate
                            action_sequence.append((name, best_params))
                    
                    else:
                        try:
                            candidate_grid = fn(candidate_grid)
                            action_sequence.append((name, params if params else {}))
                        except Exception as e:
                            self.logger.debug(f"Error executing {name}: {e}")
                            continue
                
                # Evaluate final result
                valence = amc.evaluate_valence(candidate_grid, 0.0, 0.0, si=si)
                if log:
                    self.logger.debug(f"ME recall: tried {action_sequence} (valence {valence:.2f})")
                
                if valence > best_valence:
                    best_valence = valence
                    best_grid = candidate_grid
                    best_action_seq = list(action_sequence)
                    
            except Exception as e:
                self.logger.debug(f"Error processing transformation: {e}")
                continue
        
        return best_valence, best_grid, best_action_seq

    def interpolate_in_latent(self, grid1, grid2, alpha=0.5):
        """Interpolate between two grids in latent space"""
        z1 = self.encode(grid1)
        z2 = self.encode(grid2)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Determine output shape
        h = max(grid1.shape[0], grid2.shape[0])
        w = max(grid1.shape[1], grid2.shape[1])
        
        return self.decode(z_interp, (h, w)).numpy()

    def find_transformation_in_latent(self, input_grid, output_grid):
        """Learn transformation vector in latent space"""
        z_in = self.encode(input_grid)
        z_out = self.encode(output_grid)
        transformation_vector = z_out - z_in
        return transformation_vector.detach().numpy()

    def apply_latent_transformation(self, grid, transformation_vector):
        """Apply learned transformation in latent space"""
        z = self.encode(grid)
        z_transformed = z + torch.tensor(transformation_vector, dtype=torch.float32)
        return self.decode(z_transformed, grid.shape).numpy()

    def minimize_prediction_error(self, observed_grid, expected_grid, max_iterations=10):
        """
        Minimize prediction error through active inference.
        Aligns with FEP principles from the paper.
        """
        current_grid = observed_grid.copy()
        
        for iteration in range(max_iterations):
            # Encode both grids
            z_current = self.encode(current_grid)
            z_expected = self.encode(expected_grid)
            
            # Compute prediction error in latent space
            prediction_error = z_expected - z_current
            error_magnitude = torch.norm(prediction_error).item()
            
            if error_magnitude < 0.1:  # Convergence threshold
                break
            
            # Generate action to reduce error
            # Try each skill and select the one that reduces error most
            best_skill = None
            best_params = None
            best_error = error_magnitude
            
            for skill_name, skill_fn in self.skills.items():
                if skill_name in ["color_shift", "object_fill"]:
                    # Try different parameters
                    for param in range(1, self.num_colors):
                        try:
                            if skill_name == "color_shift":
                                candidate = skill_fn(current_grid, shift=param, num_colors=self.num_colors)
                            else:
                                candidate = skill_fn(current_grid, fill_color=param)
                            
                            z_candidate = self.encode(candidate)
                            new_error = torch.norm(z_expected - z_candidate).item()
                            
                            if new_error < best_error:
                                best_error = new_error
                                best_skill = skill_name
                                best_params = {"shift": param} if skill_name == "color_shift" else {"fill_color": param}
                        except:
                            continue
                else:
                    try:
                        candidate = skill_fn(current_grid)
                        z_candidate = self.encode(candidate)
                        new_error = torch.norm(z_expected - z_candidate).item()
                        
                        if new_error < best_error:
                            best_error = new_error
                            best_skill = skill_name
                            best_params = {}
                    except:
                        continue
            
            # Apply best action
            if best_skill:
                current_grid = self.execute(current_grid, best_skill, best_params)
                self.logger.info(f"ME: Applied {best_skill} to reduce error from {error_magnitude:.3f} to {best_error:.3f}")
            else:
                break
        
        return current_grid

    def execute_with_ethical_feedback(self, grid, skill_name, params, amc, si):
        """
        Execute skill with ethical feedback from AMC module.
        Can modify or veto actions based on ethical considerations.
        """
        # Get ethical feedback before execution
        if hasattr(amc, 'generate_ethical_feedback_signal'):
            feedback = amc.generate_ethical_feedback_signal(
                proposed_action=(skill_name, params),
                current_state=grid,
                si=si
            )
            
            if feedback["signal"] == "inhibit":
                self.logger.warning(f"ME: Action {skill_name} inhibited due to {feedback['reason']}")
                
                # Try alternative if suggested
                if "alternative_bias" in feedback:
                    alt_skill, alt_params = feedback["alternative_bias"]
                    self.logger.info(f"ME: Trying alternative {alt_skill}")
                    return self.execute(grid, alt_skill, alt_params)
                else:
                    return grid  # Return unchanged
            
            elif feedback["signal"] == "enhance":
                self.logger.info(f"ME: Action {skill_name} enhanced due to positive ethical prediction")
        
        # Execute the action
        result = self.execute(grid, skill_name, params)
        
        # Post-execution ethical check
        post_valence = amc.evaluate_valence(result, si=si)
        if post_valence < -0.5:
            self.logger.warning(f"ME: Post-execution ethical check failed (valence={post_valence:.2f})")
        
        return result

    def execute_sequence(self, grid, action_sequence, amc=None, si=None):
        """
        Execute a sequence of skills with intermediate ethical checks.
        """
        current_grid = grid.copy()
        execution_log = []
        
        for i, (skill_name, params) in enumerate(action_sequence):
            # Pre-execution check if AMC available
            if amc and si:
                valence = amc.evaluate_valence(current_grid, si=si)
                if amc.veto(valence, si):
                    self.logger.warning(f"ME: Sequence halted at step {i} due to veto")
                    break
            
            # Execute skill
            current_grid = self.execute(current_grid, skill_name, params)
            execution_log.append({
                "step": i,
                "skill": skill_name,
                "params": params,
                "result_shape": current_grid.shape,
                "non_zero_count": np.count_nonzero(current_grid)
            })
        
        return current_grid, execution_log

    def learn_skill_composition(self, input_grid, output_grid, si):
        """
        Learn a composition of skills that transforms input to output.
        Uses beam search with AMC valence guidance.
        """
        beam_width = 5
        max_depth = 5
        
        # Initialize beam with input grid
        beam = [(input_grid, [], 0.0)]  # (grid, action_sequence, cost)
        
        for depth in range(max_depth):
            new_beam = []
            
            for current_grid, action_seq, cost in beam:
                # Check if we've reached the target
                if np.array_equal(current_grid, output_grid):
                    return action_seq
                
                # Try each skill
                for skill_name, skill_fn in self.skills.items():
                    if skill_name in ["color_shift", "object_fill"]:
                        # Try a few parameter values
                        for param in [1, 2, 3, 5, 7]:
                            params = {"shift": param, "num_colors": self.num_colors} if skill_name == "color_shift" else {"fill_color": param}
                            try:
                                candidate = self.execute(current_grid, skill_name, params)
                                
                                # Compute similarity to target
                                similarity = 1 - np.mean(np.abs(candidate - output_grid))
                                new_cost = cost - similarity  # Negative because we minimize
                                
                                new_beam.append((
                                    candidate,
                                    action_seq + [(skill_name, params)],
                                    new_cost
                                ))
                            except:
                                continue
                    else:
                        try:
                            candidate = skill_fn(current_grid)
                            similarity = 1 - np.mean(np.abs(candidate - output_grid))
                            new_cost = cost - similarity
                            
                            new_beam.append((
                                candidate,
                                action_seq + [(skill_name, {})],
                                new_cost
                            ))
                        except:
                            continue
            
            # Keep top beam_width candidates
            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]
        
        # Return best sequence found
        return beam[0][1] if beam else []
