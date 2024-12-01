from pathlib import Path

from deform_rl.algos.save_manager import reset_manager, get_paths


try:
    cur_dir = Path(__file__).parent

    reset_manager(cur_dir.parent/"logs", cur_dir.parent /
                  "saved_models", cur_dir.parent/"saved_norms")
except ValueError:
    print("Manager already set up")
