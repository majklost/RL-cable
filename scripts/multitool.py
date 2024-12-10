from argparse import ArgumentParser
from pathlib import Path
from deform_rl.algos.save_manager import print_experiment, load_manager

"""
Many small functions for set up of broken things
"""


def perform_print():
    EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
    load_manager(EXPERIMENTS_PATH)

    parser = ArgumentParser(prog="print_experiment",
                            description="Print the experiment details.")
    parser.add_argument("experiment_name", type=str)

    print_experiment(parser.parse_args().experiment_name)
