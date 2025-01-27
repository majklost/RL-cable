from pathlib import Path

from deform_rl.algos.save_manager import reset_manager, get_paths, delete_experiment


# delete_experiment('tiny_experiment')
# delete_experiment('rect2D')

# cur_dir = Path(__file__).parent

# reset_manager(cur_dir.parent/"logs", cur_dir.parent /
#               "saved_models", cur_dir.parent/"saved_norms")


# first_paths = get_paths("tiny_experiment", "just testing", continue_run=False)
# with open(first_paths['model_last'], 'w') as f:
#     f.write("Hello, previous!")

# second_paths = get_paths("tiny_experiment", "just testing", continue_run=True)
# with open(second_paths['model_last'], 'w') as f:
#     f.write("Hello, again!")
