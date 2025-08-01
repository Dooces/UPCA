import json
import numpy as np
import logging
import glob
import os
from me_engine import MEEngine
from ma_engine import MAEngine
from amc_module import AMCModule
from si_scaffold import SIScaffold
from upca_agent import UPCAAgent

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("main_arc")

def pad_to_shape(arr, shape):
    arr = np.array(arr)
    pad_height = shape[0] - arr.shape[0]
    pad_width = shape[1] - arr.shape[1]
    pad_before = (0, 0)
    pad_after = (max(pad_height, 0), max(pad_width, 0))
    pad_widths = ((0, pad_after[0]), (0, pad_after[1]))
    return np.pad(arr, pad_widths, mode='constant', constant_values=0)[:shape[0], :shape[1]]

def calculate_similarity(predicted_output, target_output, si=None):
    if si is not None:
        return si.robust_similarity_fn(predicted_output, target_output)
    if isinstance(predicted_output, tuple):
        predicted_output = predicted_output[0]
    if isinstance(target_output, tuple):
        target_output = target_output[0]
    target_shape = (
        max(predicted_output.shape[0], target_output.shape[0]),
        max(predicted_output.shape[1], target_output.shape[1])
    )
    pred_padded = pad_to_shape(predicted_output, target_shape)
    out_padded = pad_to_shape(target_output, target_shape)
    return np.mean(pred_padded == out_padded)

# --- 1. Instantiate UPCA modules ---
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

# --- 2. Find all ARC .json files in the training directory ---
arc_dir = "ARC-AGI-2/data/evaluation/"
arc_task_paths = sorted(glob.glob(os.path.join(arc_dir, "*.json")))

logger.info(f"Found {len(arc_task_paths)} ARC tasks.")

# --- 3. Track overall success/failure ---
total_tests = 0
total_successes = 0
total_failures = 0

# --- 4. Test on all ARC tasks ---
for arc_task_path in arc_task_paths:
    try:
        with open(arc_task_path) as f:
            task = json.load(f)
    except Exception as e:
        logger.warning(f"Skipping {arc_task_path}: {e}")
        continue
    logger.info(f"\n=== ARC Task: {arc_task_path} ===")

    # --- Train on train pairs (agent must discover, not brute-force) ---
    for idx, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        predicted_output, agent_action_seq, valence = agent.solve_arc(input_grid, output_grid)
        similarity = calculate_similarity(predicted_output, output_grid, si)
        logger.info(f"Train {idx}: similarity={similarity:.2f}, action_seq={agent_action_seq}")
        if similarity > 0.95:
            si.store_transformation(input_grid, agent_action_seq, predicted_output, generalize=True)
            logger.info(f"Stored general skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, 1.0, predicted_output)
        elif similarity > 0.5:
            si.store_transformation(input_grid, agent_action_seq, predicted_output, generalize=False)
            logger.info(f"Stored partial skill sequence: {agent_action_seq}")
            agent.update(input_grid, agent_action_seq, similarity, predicted_output)
        else:
            agent.update(input_grid, agent_action_seq, -1.0, predicted_output)
        agent.observe_example(input_grid, output_grid)

    # --- Test on test pairs ---
    logger.info(f"\n--- Testing on test pairs for {arc_task_path} ---")
    for idx, pair in enumerate(task['test']):
        test_input = np.array(pair['input'])
        target_output = np.array(pair['output'])
        predicted_output, agent_action_seq, valence = agent.solve_arc(test_input, target_output)
        similarity = calculate_similarity(predicted_output, target_output, si)
        logger.info(f"Test {idx}: similarity={similarity:.2f}, action_seq={agent_action_seq}")
        logger.info(f"Agent prediction:\n{predicted_output}")
        logger.info(f"Ground truth:\n{target_output}")
        total_tests += 1
        if similarity > 0.95:
            logger.info("Correct.")
            total_successes += 1
        else:
            logger.info("Incorrect.")
            total_failures += 1

    # --- Meta-learn and log after each ARC task ---
    agent.amc.meta_learn_weights(agent.si)
    logger.info(f"AMC weights after meta-learning: learning={agent.amc.learning_weight:.2f}, success={agent.amc.success_weight:.2f}")
    logger.info(f"Learning log (last 10): {[entry for entry in si.learning_log[-10:]]}")

si.save("si_scaffold_arc.pkl")
logger.info(f"\nTesting complete. SI stored transformations: {len(si.transformations)}")
for trans in si.transformations[:10]:
    logger.info(f"Action sequence: {trans['action_sequence']}")
    logger.info(f"Input pattern:\n{trans['input_pattern']}")
    logger.info(f"Output pattern:\n{trans['output_pattern']}")

# --- 5. Output quantitative score ---
logger.info(f"\n=== ARC-AGI Quantitative Score ===")
logger.info(f"Total test cases: {total_tests}")
logger.info(f"Total correct: {total_successes}")
logger.info(f"Total incorrect: {total_failures}")
if total_tests > 0:
    logger.info(f"Overall accuracy: {100.0 * total_successes / total_tests:.2f}%")
else:
    logger.info("No test cases run.")
