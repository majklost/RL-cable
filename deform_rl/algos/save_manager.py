# Helper for saving and loading models and VecNormalizers.
import pickle
import datetime
from pathlib import Path

# No need for object now


class _Experiment:
    def __init__(self):
        self.run_cnt = 0
        self.comment = []

    def save_comments(self, fpath):
        with open(fpath, 'w') as f:
            for c in enumerate(self.comment):
                f.write(f"{c[0]}. {c[1]}\n")


class _SaveManager:
    """
    For each experiment name, it creates directories for tensorboard logs, models, and VecNormalizers.
    When get_experiment is called, it returns path even with filenames
    """

    def __init__(self, tb_log_dir: Path | str, model_dir: Path | str, vec_norm_dir: Path | str):
        self.tb_log_dir = Path(tb_log_dir).absolute()
        self.model_dir = Path(model_dir).absolute()
        self.vec_norm_dir = Path(vec_norm_dir).absolute()
        self.experiments = {}

    def get_paths(self, experiment_name: str, comment: str, continue_run: bool = False) -> dict[str, Path]:
        """
        returns paths for tensorboard logs, models, and VecNormalizers
        if experiment_name is new, it creates new directories
        :param experiment_name: (str) the name of the experiment
        :param comment: (str) the comment for the experiment
        :param continue_run: (bool) whether to count it as a new run or continue the previous one

        :return: (dict) the paths for tensorboard logs, models, and VecNormalizers
        """
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = _Experiment()
            self._create_folders(experiment_name)
        experiment = self.experiments[experiment_name]
        if continue_run:
            assert experiment.run_cnt > 0, "No previous run to continue"
        if not continue_run:
            experiment.run_cnt += 1
            experiment.comment.append(comment)
            if experiment.run_cnt % 5:
                experiment.save_comments(
                    self.model_dir / experiment_name / "comments.txt")
        else:
            experiment.comment[-1] += " || " + comment

        self.backup()
        return {
            "tb": self.tb_log_dir / experiment_name / (experiment_name+"_tb"),
            "model_last": self.model_dir / experiment_name / create_last_model_fname(experiment_name, experiment.run_cnt),
            "model_best": self.vec_norm_dir / experiment_name / create_best_model_fname(experiment_name, experiment.run_cnt),
            "norm": self.vec_norm_dir / experiment_name / (create_fname(experiment_name, experiment.run_cnt) + ".pkl")
        }

    def force_comments(self):
        """
        Force comments to be saved.
        """
        for experiment_name, experiment in self.experiments.items():
            experiment.save_comments(
                self.model_dir / experiment_name / "comments.txt")

    def _create_folders(self, experiment_name: str):
        """
        Create folders for tensorboard logs, models, and VecNormalizers.
        """
        tb_log_dir = self.tb_log_dir / experiment_name
        model_dir = self.model_dir / experiment_name
        vec_norm_dir = self.vec_norm_dir / experiment_name

        tb_log_dir.mkdir(parents=True, exist_ok=False)
        model_dir.mkdir(parents=True, exist_ok=False)
        vec_norm_dir.mkdir(parents=True, exist_ok=False)

        (model_dir / 'comments.txt').touch()

        return tb_log_dir, model_dir, vec_norm_dir

    def backup(self):
        """
        Backup the SaveManager object.
        """
        with open(Path(__file__).parent/"save_manager.pkl", "wb") as f:
            pickle.dump(self, f)

    def consistency_check(self):
        """
        Check if the directories are consistent with the SaveManager object.
        """
        for experiment_name, experiment in self.experiments.items():
            tb_log_dir = self.tb_log_dir / experiment_name
            model_dir = self.model_dir / experiment_name
            vec_norm_dir = self.vec_norm_dir / experiment_name
            assert tb_log_dir.exists(), f"{tb_log_dir} does not exist"
            assert model_dir.exists(), f"{model_dir} does not exist"
            assert vec_norm_dir.exists(), f"{vec_norm_dir} does not exist"

    def clean_keys(self):
        """
        If any of the keys are not in the directories, remove them.
        """
        keys = list(self.experiments.keys())
        for key in keys:
            tb_log_dir = self.tb_log_dir / key
            model_dir = self.model_dir / key
            vec_norm_dir = self.vec_norm_dir / key
            if not tb_log_dir.exists() or not model_dir.exists() or not vec_norm_dir.exists():
                del self.experiments[key]
        self.backup()

    def __str__(self):
        return f"SaveManager(tb_log_dir={self.tb_log_dir}, model_dir={self.model_dir}, vec_norm_dir={self.vec_norm_dir})"


def get_datetime_str():
    return datetime.datetime.now().strftime("%d-%m-%H-%M-%S")


def create_fname(experiment_name: str, run_cnt: int):
    return f"{experiment_name}_r{run_cnt}_{get_datetime_str()}"


def create_last_model_fname(experiment_name: str, run_cnt: int):
    return create_fname(experiment_name, run_cnt) + "_last_model"


def create_best_model_fname(experiment_name: str, run_cnt: int):
    return create_fname(experiment_name, run_cnt) + "_best_model"


try:
    manager = pickle.load(open(Path(__file__).parent/"save_manager.pkl", "rb"))
except FileNotFoundError:
    print("No save_manager.pkl found. Create one with reset_manager function.")
    manager = None


def reset_manager(tb_log_dir: Path, model_dir: Path, vec_norm_dir: Path):
    global manager
    if manager is not None:
        raise ValueError(
            "SaveManager already exists. Delete or rename it and all it's content before creating a new one.")
    manager = _SaveManager(tb_log_dir, model_dir, vec_norm_dir)
    manager.backup()
    print("SaveManager reseted.")


def get_paths(experiment_name: str, comment: str, continue_run: bool = False) -> dict[str, Path]:
    return manager.get_paths(experiment_name, comment, continue_run)


def force_comments():
    manager.force_comments()


def consistency_check():
    manager.consistency_check()


def clean_keys():
    manager.clean_keys()
