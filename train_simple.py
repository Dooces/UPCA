import numpy as np
import logging
import time
import random
from me_engine import MEEngine
from ma_engine import MAEngine
from amc_module import AMCModule
from si_scaffold import SIScaffold
from upca_agent import UPCAAgent

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("train_simple")

max_minutes = 1
max_seconds = max_minutes * 60
start_time = time.time()

me = MEEngine(latent_dim=16, num_colors=10)
ma = MAEngine(me)
amc = AMCModule()
si = SIScaffold(me_engine=me)
agent = UPCAAgent(
    me, ma, amc, si,
    log_level=logging.INFO,
    me_enabled=True,
    ma_enabled=True,
    me_efficiency=1.0,
    ma_efficiency=1.0
)

min_grid_size = 2
max_grid_size = 15
num_colors = 4
tasks_per_level = 10
test_tasks_per_level = 5

def pad_to_shape(arr, shape, pad_value=0):
    arr = np.array(arr)
    # If arr is a list of arrays, use the first one
    if isinstance(arr, list) or arr.ndim > 2:
        arr = np.array(arr[0]) if len(arr) > 0 else np.zeros(shape, dtype=int)
    if arr.ndim != 2:
        arr = arr.reshape((arr.size, 1)) if arr.size > 0 else np.zeros(shape, dtype=int)
    h, w = arr.shape
    pad_height = shape[0] - h
    pad_width = shape[1] - w
    pad_before = (0, 0)
    pad_after = (max(pad_height, 0), max(pad_width, 0))
    pad_widths = ((0, pad_after[0]), (0, pad_after[1]))
    return np.pad(arr, pad_widths, mode='constant', constant_values=pad_value)[:shape[0], :shape[1]]

def random_grid(size, num_colors=4, density=0.3):
    grid = np.zeros((size, size), dtype=int)
    mask = np.random.rand(size, size) < density
    grid[mask] = np.random.randint(1, num_colors, size=mask.sum())
    return grid

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

# Add missing skills to ME engine if needed
if "object_fill" not in me.skills:
    me.skills["object_fill"] = object_fill
if "largest_object_crop" not in me.skills:
    me.skills["largest_object_crop"] = largest_object_crop

# Get list of available transformations from ME engine
available_transforms = list(me.skills.values())
transform_names = list(me.skills.keys())

def random_transform(me_engine, input_grid):
    num_steps = np.random.choice([1, 2, 3, 4])
    transform_indices = np.random.choice(len(available_transforms), size=num_steps, replace=True)
    candidate = input_grid.copy()
    action_sequence = []
    
    for idx in transform_indices:
        fn = available_transforms[idx]
        fn_name = transform_names[idx]
        
        if fn_name == "color_shift":
            shift = np.random.randint(1, num_colors)
            candidate = me_engine.execute(candidate, fn_name, {"shift": shift, "num_colors": num_colors})
            action_sequence.append((fn_name, {"shift": shift, "num_colors": num_colors}))
        elif fn_name == "object_fill":
            fill_color = np.random.randint(1, num_colors)
            candidate = me_engine.execute(candidate, fn_name, {"fill_color": fill_color})
            action_sequence.append((fn_name, {"fill_color": fill_color}))
        else:
            candidate = me_engine.execute(candidate, fn_name, {})
            action_sequence.append((fn_name, {}))
    
    return candidate, action_sequence

def compositional_tasks(me_engine, grid_size, num_colors, min_steps=2, max_steps=5):
    tasks = []
    for _ in range(10):
        input_grid = random_grid(grid_size, num_colors)
        num_steps = np.random.randint(min_steps, max_steps+1)
        transform_indices = np.random.choice(len(available_transforms), size=num_steps, replace=True)
        candidate = input_grid.copy()
        action_sequence = []
        
        for idx in transform_indices:
            fn_name = transform_names[idx]
            
            if fn_name == "color_shift":
                shift = np.random.randint(1, me_engine.num_colors)
                candidate = me_engine.execute(candidate, fn_name, {"shift": shift, "num_colors": me_engine.num_colors})
                action_sequence.append((fn_name, {"shift": shift, "num_colors": me_engine.num_colors}))
            elif fn_name == "object_fill":
                fill_color = np.random.randint(1, me_engine.num_colors)
                candidate = me_engine.execute(candidate, fn_name, {"fill_color": fill_color})
                action_sequence.append((fn_name, {"fill_color": fill_color}))
            else:
                candidate = me_engine.execute(candidate, fn_name, {})
                action_sequence.append((fn_name, {}))

        output_grid = candidate
        tasks.append((input_grid, output_grid, action_sequence))
    return tasks

def calculate_similarity(predicted_output, target_output, si=None):
    # Ensure both are 2D numpy arrays
    def to_grid(arr):
        if isinstance(arr, list):
            # If it's a list of arrays, use the first one
            if len(arr) == 0:
                return np.zeros((1, 1), dtype=int)
            arr = np.array(arr[0]) if isinstance(arr[0], (np.ndarray, list)) else np.array(arr)
        if not hasattr(arr, "shape"):
            arr = np.array(arr)
        if arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            arr = arr.reshape((int(np.sqrt(arr.size)), -1)) if arr.size > 0 else np.zeros((1, 1), dtype=int)
        return arr

    predicted_output = to_grid(predicted_output)
    target_output = to_grid(target_output)

    if si is not None:
        return si.robust_similarity_fn(predicted_output, target_output)
    target_shape = (
        max(predicted_output.shape[0], target_output.shape[0]),
        max(predicted_output.shape[1], target_output.shape[1])
    )
    pred_padded = pad_to_shape(predicted_output, target_shape)
    out_padded = pad_to_shape(target_output, target_shape)
    return np.mean(pred_padded == out_padded)

def test_agent(agent, grid_size, num_colors, num_tests=5):
    successes = 0
    failures = []
    for i in range(num_tests):
        input_grid = random_grid(grid_size, num_colors)
        output_grid, _ = random_transform(agent.me, input_grid)
        
        # Create examples for MA to learn from
        examples = [(input_grid, output_grid)]
        
        predicted_output, agent_action_seq, valence = agent.solve_arc(
            input_grid, output_grid, examples=examples
        )
        similarity = calculate_similarity(predicted_output, output_grid)
        if similarity > 0.8:
            logger.info(f"Test {i}: Agent produced output (success, similarity={similarity:.2f})")
            successes += 1
        else:
            logger.info(f"Test {i}: Agent failed (similarity={similarity:.2f})")
            failures.append(input_grid)
    return successes / num_tests, failures

dream_on_failure = True
dream_every_n_levels = 3
level_count = 0
consecutive_failures = 0

def dynamic_efficiency_update(agent, avg_progress, avg_valence):
    if avg_progress > 0.2 and avg_valence > 0.8:
        agent.me_efficiency = 0.1
        agent.ma_efficiency = 1.0
        logger.info(f"[BOSS] MA dominates (me_efficiency={agent.me_efficiency}, ma_efficiency={agent.ma_efficiency})")
    elif avg_progress < 0.05 or avg_valence < 0.5:
        agent.me_efficiency = 1.0
        agent.ma_efficiency = 0.5
        logger.info(f"[BOSS'S BOSS] ME helps recover (me_efficiency={agent.me_efficiency}, ma_efficiency={agent.ma_efficiency})")
    else:
        agent.me_efficiency = 0.5
        agent.ma_efficiency = 0.8
        logger.info(f"[TEAM] Balanced (me_efficiency={agent.me_efficiency}, ma_efficiency={agent.ma_efficiency})")

grid_size = min_grid_size
while time.time() - start_time < max_seconds:
    logger.info(f"\n=== Training at grid size {grid_size}x{grid_size} ===")
    successes = 0

    comp_tasks = compositional_tasks(me, grid_size, num_colors)
    for input_grid, output_grid, true_seq in comp_tasks:
        # Create examples for MA to learn from
        examples = [(input_grid, output_grid)]
        
        predicted_output, agent_action_seq, valence = agent.solve_arc(
            input_grid, output_grid, examples=examples
        )
        similarity = calculate_similarity(predicted_output, output_grid)
        logger.info(f"Compositional task similarity: {similarity:.2f}")
        si.log_learning(
            step=level_count,
            progress=similarity,
            success=(1.0 if similarity > 0.95 else 0.0),
            valence=valence
        )
        if similarity > 0.95:
            si.store_transformation(input_grid, agent_action_seq, output_grid, generalize=True)
            logger.info(f"Stored general skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, 1.0, output_grid)
            successes += 1
        elif similarity > 0.5:
            si.store_transformation(input_grid, agent_action_seq, output_grid, generalize=False)
            logger.info(f"Stored partial skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, similarity, output_grid)
        else:
            agent.update(input_grid, agent_action_seq, -1.0, output_grid)
        agent.observe_example(input_grid, output_grid)
        logger.debug(f"Train input:\n{input_grid}\nTrain output:\n{output_grid}")

    for task_num in range(tasks_per_level):
        if time.time() - start_time > max_seconds:
            logger.info("Time limit reached. Stopping training.")
            break
        input_grid = random_grid(grid_size, num_colors)
        output_grid, true_seq = random_transform(me, input_grid)
        
        # Create examples for MA to learn from
        examples = [(input_grid, output_grid)]
        
        predicted_output, agent_action_seq, valence = agent.solve_arc(
            input_grid, output_grid, examples=examples
        )
        similarity = calculate_similarity(predicted_output, output_grid)
        logger.info(f"Random task similarity: {similarity:.2f}")
        if similarity > 0.95:
            si.store_transformation(input_grid, agent_action_seq, output_grid, generalize=True)
            logger.info(f"Stored general skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, 1.0, output_grid)
            successes += 1
        elif similarity > 0.5:
            si.store_transformation(input_grid, agent_action_seq, output_grid, generalize=False)
            logger.info(f"Stored partial skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, similarity, output_grid)
        else:
            agent.update(input_grid, agent_action_seq, -1.0, output_grid)
        agent.observe_example(input_grid, output_grid)
        logger.debug(f"Train input:\n{input_grid}\nTrain output:\n{output_grid}")

    success_rate = successes / (tasks_per_level + len(comp_tasks))
    logger.info(f"Success rate at grid size {grid_size}: {success_rate:.2f}")

    logger.info(f"Testing agent after training at grid size {grid_size}")
    test_success_rate, failures = test_agent(agent, grid_size, num_colors, num_tests=test_tasks_per_level)
    logger.info(f"Test success rate at grid size {grid_size}: {test_success_rate:.2f}")

    agent.amc.meta_learn_weights(agent.si)
    logger.info(f"AMC weights after meta-learning: learning={agent.amc.learning_weight:.2f}, success={agent.amc.success_weight:.2f}")

    if test_success_rate < 1.0:
        consecutive_failures += 1
    else:
        consecutive_failures = 0

    dream_now = False
    if dream_on_failure and consecutive_failures >= 3:
        dream_now = True
        logger.info(f"Dreaming triggered by 3 consecutive failures at grid size {grid_size}")
        consecutive_failures = 0
    elif (level_count + 1) % dream_every_n_levels == 0:
        dream_now = True
        logger.info(f"Dreaming triggered by curriculum interval at grid size {grid_size}")

    if dream_now:
        agent.dream()

    level_count += 1

    avg_progress = np.mean([entry["progress"] for entry in si.learning_log[-10:]]) if len(si.learning_log) >= 10 else 0
    avg_valence = np.mean([entry["valence"] for entry in si.learning_log[-10:]]) if len(si.learning_log) >= 10 else 0
    dynamic_efficiency_update(agent, avg_progress, avg_valence)
    if agent.amc.ready_to_escalate(agent.si, window=10, min_progress=0.05, min_valence=0.7):
        logger.info("AMC: Not enough learning progress or valence, escalating to harder tasks.")
    else:
        logger.info("AMC: Still learning/generalizing, keep training at current level.")

    logger.info(f"Learning log (last 10): {[entry for entry in si.learning_log[-10:]]}")

    grid_size += 1
    if grid_size > max_grid_size:
        grid_size = min_grid_size

si.save("si_scaffold_self_escalate.pkl")
elapsed = time.time() - start_time
logger.info(f"\nTraining complete in {elapsed/60:.2f} minutes. SI stored transformations: {len(si.transformations)}")
for trans in si.transformations[:10]:
    logger.info(f"Action sequence: {trans['action_sequence']}")
    logger.info(f"Input pattern:\n{trans['input_pattern']}")
    logger.info(f"Output pattern:\n{trans['output_pattern']}")
