from pathlib import Path

from deform_rl.algos.save_manager import move_dirs, get_paths


try:
    cur_dir = Path(__file__).parent

    move_dirs(cur_dir.parent/"experiments"/"logs", cur_dir.parent / "experiments" /
              "saved_models", cur_dir.parent/"experiments"/"saved_norms")
except ValueError:
    print("Manager already set up")
