from pathlib import Path

from deform_rl.algos.save_manager import reset_manager, get_paths

cur_dir = Path(__file__).parent

# reset_manager(cur_dir.parent/"logs", cur_dir.parent /
#               "saved_models", cur_dir.parent/"saved_norms")
paths = get_paths("tiny_experiment", "just testing", continue_run=True)
print(paths)
