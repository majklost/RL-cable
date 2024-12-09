# Helper for saving and loading models and VecNormalizers.
import pickle
import datetime
from pathlib import Path
from typing import TypedDict
import shutil
import warnings
# No need for object now


class PathsDict(TypedDict):
    tb: Path
    model_last: Path
    model_best: Path
    norm: Path
    env_name: str
    data: dict


class _Experiment:
    def __init__(self):
        self.run_cnt = 0
        self.comment = []
        self.env_name = None
        # self.last_date = datetime.datetime.now()
        self.dates = []
        self.data = {}

    def save_comments(self, fpath):
        with open(fpath, 'w') as f:
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Total runs: {self.run_cnt}\n")
            for k, v in self.data.items():
                f.write(f"{k}: {v}\n")
            for c in enumerate(self.comment):
                f.write(f"{c[0]}. {c[1]}\n")

    def __str__(self):
        return f"Experiment(run_cnt={self.run_cnt}, comment={self.comment}, env_name={self.env_name}, dates={self.dates}, data={self.data})"


class _SaveManager:
    """
    For each experiment name, it creates directories for tensorboard logs, models, and VecNormalizers.
    When get_experiment is called, it returns path even with filenames
    """

    def __init__(self, tb_log_dir: Path | str, model_dir: Path | str, vec_norm_dir: Path | str):
        self.tb_log_dir = Path(tb_log_dir)
        self.model_dir = Path(model_dir)
        self.vec_norm_dir = Path(vec_norm_dir)
        self.experiments = {}

    def move_dirs(self, tb_log_dir: Path | str, model_dir: Path | str, vec_norm_dir: Path | str):
        olds = [self.tb_log_dir, self.model_dir, self.vec_norm_dir]
        news = [Path(tb_log_dir).absolute(),
                Path(model_dir).absolute(), Path(vec_norm_dir).absolute()]
        for old, new in zip(olds, news):
            shutil.move(old, new)
        self.tb_log_dir, self.model_dir, self.vec_norm_dir = news
        self.backup()

    def get_paths(self, experiment_name: str, comment: str, env_name: str, continue_run: bool = False, data={}) -> dict[str, Path]:
        """
        returns paths for tensorboard logs, models, and VecNormalizers
        if experiment_name is new, it creates new directories
        :param experiment_name: (str) the name of the experiment
        :param comment: (str) the comment for the experiment
        :param continue_run: (bool) whether to count it as a new run or continue the previous one
        :param data: (dict) additional data to save

        :return: (dict) the paths for tensorboard logs, models, and VecNormalizers
            -- tb: (Path) the path to tensorboard logs
            -- model_last: (Path) the path to the last model
            -- model_best: (Path) the path to the best model
            -- norm: (Path) the path to the VecNormalizer
        """
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = _Experiment()
            self._create_folders(experiment_name)
        experiment = self.experiments[experiment_name]
        if experiment.env_name is None:
            experiment.env_name = env_name
        elif experiment.env_name != env_name:
            warnings.warn(
                f"Environment name changed from {experiment.env_name} to {env_name}")
            experiment.env_name = env_name

        if continue_run:
            assert experiment.run_cnt > 0, "No previous run to continue"
        if not continue_run:
            experiment.run_cnt += 1
            experiment.dates.append(datetime.datetime.now())
            experiment.comment.append(comment)
            if experiment.run_cnt % 5:
                experiment.save_comments(
                    self.model_dir / experiment_name / "comments.txt")
        else:
            experiment.comment[-1] += " || " + comment
        experiment.data = data

        data = getattr(experiment, 'data', {})
        if not data:
            warnings.warn("No data found in the experiment.")

        self.backup()
        return {
            "tb": self.tb_log_dir / experiment_name / create_fname(experiment_name, experiment.run_cnt, experiment.dates[-1]),
            "model_last": self.model_dir / experiment_name / create_last_model_fname(experiment_name+"_last", experiment.run_cnt, experiment.dates[-1]),
            "model_best": self.model_dir / experiment_name / create_best_model_fname(experiment_name+'_best', experiment.run_cnt, experiment.dates[-1]),
            "norm": self.vec_norm_dir / experiment_name / (create_fname(experiment_name, experiment.run_cnt, experiment.dates[-1]) + ".pkl"),
            "env_name": experiment.env_name,
            "data": data
        }

    def get_run_paths(self, experiment_name: str, run_cnt: int = -1):
        """
        Get the paths for a specific run of an experiment.
        """
        try:
            experiment = self.experiments[experiment_name]
            print(experiment)
        except KeyError:
            raise ValueError(
                f"{experiment_name} not found in the experiments, current experiments are {list(self.experiments.keys())}")

        if run_cnt == -1:
            run_cnt = experiment.run_cnt

        data = getattr(experiment, 'data', {})
        if not data:
            warnings.warn("No data found in the experiment.")

        return {
            "tb": self.tb_log_dir / experiment_name / create_fname(experiment_name, run_cnt, experiment.dates[run_cnt-1]),
            "model_last": self.model_dir / experiment_name / create_last_model_fname(experiment_name+"_last", run_cnt, experiment.dates[run_cnt-1]),
            "model_best": self.model_dir / experiment_name / create_best_model_fname(experiment_name+'_best', run_cnt, experiment.dates[run_cnt-1]),
            "norm": self.vec_norm_dir / experiment_name / (create_fname(experiment_name, run_cnt, experiment.dates[run_cnt-1]) + ".pkl"),
            "env_name": experiment.env_name,
            "data": data
        }

    def forget_last_run(self, experiment_name: str):
        """
        Delete the last run of the experiment.
        """
        experiment = self.experiments[experiment_name]
        run_cnt = experiment.run_cnt
        experiment.run_cnt -= 1
        experiment.dates.pop()
        experiment.comment.pop()
        self.backup()

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

    def delete_experiment(self, experiment_name: str):
        """
        Delete the experiment.
        """
        tb_log_dir = self.tb_log_dir / experiment_name
        model_dir = self.model_dir / experiment_name
        vec_norm_dir = self.vec_norm_dir / experiment_name
        shutil.rmtree(model_dir)
        shutil.rmtree(tb_log_dir)
        shutil.rmtree(vec_norm_dir)
        del self.experiments[experiment_name]
        self.backup()

    def __str__(self):
        return f"SaveManager(tb_log_dir={self.tb_log_dir}, model_dir={self.model_dir}, vec_norm_dir={self.vec_norm_dir})"


class _SaveManagerV2(_SaveManager):
    """
    Save manager with specified folder experiments, everything is in this folder
    """

    def __init__(self, experiments_folder: Path | str):
        self.experiments_folder = Path(experiments_folder)
        tb_log_dir = self.experiments_folder / "logs"
        model_dir = self.experiments_folder / "models"
        vec_norm_dir = self.experiments_folder / "norms"
        super().__init__(tb_log_dir, model_dir, vec_norm_dir)

    def backup(self):
        """
        Backup the SaveManager object.
        """
        with open(self.experiments_folder/"save_manager.pkl", "wb") as f:
            pickle.dump(self, f)

    def refresh_paths(self):
        """
        Refresh paths of the SaveManager
        """
        self.tb_log_dir = self.experiments_folder / "logs"
        self.model_dir = self.experiments_folder / "models"
        self.vec_norm_dir = self.experiments_folder / "norms"


def get_datetime_str(date: datetime.datetime):
    return date.strftime("%d-%m-%H-%M-%S")


def create_fname(experiment_name: str, run_cnt: int, date: datetime.datetime):
    return f"{experiment_name}_r{run_cnt}_{get_datetime_str(date)}"


def create_last_model_fname(experiment_name: str, run_cnt: int, date: datetime.datetime):
    return create_fname(experiment_name, run_cnt, date) + "_last_model"


def create_best_model_fname(experiment_name: str, run_cnt: int, date: datetime.datetime):
    return create_fname(experiment_name, run_cnt, date) + "_best_model"

# OLD VERSION
# try:
#     manager = pickle.load(open(Path(__file__).parent/"save_manager.pkl", "rb"))
# except FileNotFoundError:
#     print("No save_manager.pkl found. Create one with reset_manager function.")
#     manager = None


def load_manager(experiments_folder: Path | str):
    experiments_folder = Path(experiments_folder)
    global manager

    try:
        manager = pickle.load(
            open(experiments_folder/"save_manager.pkl", "rb"))
        manager.experiments_folder = experiments_folder
        manager.refresh_paths()
    except FileNotFoundError:
        print("No save_manager.pkl found. Will create a new one.")
        manager = _SaveManagerV2(experiments_folder)
    return manager


# def reset_manager(tb_log_dir: Path, model_dir: Path, vec_norm_dir: Path):
#     global manager
#     if manager is not None:
#         raise ValueError(
#             "SaveManager already exists. Delete or rename it and all it's content before creating a new one.")
#     manager = _SaveManager(tb_log_dir, model_dir, vec_norm_dir)
#     manager.backup()
#     print("SaveManager reseted.")
def reset_manager(experiments_folder: Path | str):
    global manager
    manager = _SaveManagerV2(experiments_folder)
    manager.backup()
    print("SaveManager reseted. Delete the previous one if you don't need it.")


def forget_last_run(experiment_name: str):
    manager.forget_last_run(experiment_name)


def get_paths(experiment_name: str, comment: str, env_name: str, continue_run: bool = False, data: dict = {}) -> PathsDict:
    """
    returns paths for tensorboard logs, models, and VecNormalizers
    if experiment_name is new, it creates new directories
    :param experiment_name: (str) the name of the experiment
    :param comment: (str) the comment for the experiment
    :param env_name: (str) the name of the environment
    :param continue_run: (bool) whether to count it as a new run or continue the previous one

    :return: (dict) the paths for tensorboard logs, models, and VecNormalizers
        -- tb: (Path) the path to tensorboard logs
        -- model_last: (Path) the path to the last model
        -- model_best: (Path) the path to the best model
        -- norm: (Path) the path to the VecNormalizer
        -- env_name: (str) the name of the environment
        -- data: (dict) additional data to save
    """
    return manager.get_paths(experiment_name, comment, env_name, continue_run, data)


def force_comments():
    manager.force_comments()


def consistency_check():
    manager.consistency_check()


def clean_keys():
    manager.clean_keys()


def get_run_paths(experiment_name: str, run_cnt: int = -1):
    """
        :return: (dict) the paths for tensorboard logs, models, and VecNormalizers
        -- tb: (Path) the path to tensorboard logs
        -- model_last: (Path) the path to the last model
        -- model_best: (Path) the path to the best model
        -- norm: (Path) the path to the VecNormalizer
        -- env_name: (str) the name of the environment
    """
    return manager.get_run_paths(experiment_name, run_cnt)


def delete_experiment(experiment_name: str):
    manager.delete_experiment(experiment_name)


# def move_dirs(tb_log_dir: Path | str, model_dir: Path | str, vec_norm_dir: Path | str):
#     print("Moving directories")
#     manager.move_dirs(tb_log_dir, model_dir, vec_norm_dir)
#     print("Directories moved")
