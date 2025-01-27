import optuna
from argparse import ArgumentParser
from pathlib import Path
parser = ArgumentParser()
parser.add_argument("--name", default="default_name")
args = parser.parse_args()
EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
name = args.name
study = optuna.create_study(direction='maximize', study_name=name,
                            storage=f"sqlite:///{EXPERIMENTS_PATH / 'hyperopt' / f'{name}.db'}", load_if_exists=True)
