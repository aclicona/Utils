from keras.models import load_model as load_keras_model
import os
import sys
import subprocess
import pandas as pd


def download_and_extract(
    github_repo_path, branch_zip, repo_name, branch_name, token_header, root_path
):
    repo_zip_path = os.path.join(root_path, branch_zip)

    commands = [
        ["rm", "-rf", os.path.join(root_path, repo_name)],
        ["rm", "-f", repo_zip_path],
        ["wget", "--header", token_header, github_repo_path, "-O", repo_zip_path],
        ["unzip", "-o", repo_zip_path, "-d", root_path],
        [
            "mv",
            os.path.join(root_path, f"{repo_name}-{branch_name}"),
            os.path.join(root_path, repo_name),
        ],
    ]

    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command {' '.join(command)}: {e.stderr}")


class EnvironmentDirOptions:

    def __init__(self, drive_dir: str = None, kaggle_dir: str = None) -> None:
        if drive_dir is None:
            self.root_path = "/content"
        else:
            self.root_path = drive_dir
        if kaggle_dir is None:
            self.root_path = "/kaggle/working"
        else:
            self.root_path = kaggle_dir
        if kaggle_dir is None and drive_dir is None:
            self.root_path = "."

        self.main_dir = None

    def load_google_drive_dir(self, location: str = "drive"):
        if self.is_running_in_colab():
            from google.colab import drive

            path = os.path.join(self.root_path, location)
            drive.mount(path)
            return path
        return self.root_path

    @staticmethod
    def is_running_in_colab():
        try:
            from google.colab import drive

            return True
        except ImportError:
            return False

    @staticmethod
    def is_running_in_kaggle():
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
            return True
        else:
            return False

    @staticmethod
    def save_model(model, model_name):
        model.save(model_name)

    @staticmethod
    def load_model(path):
        model = load_keras_model(path)
        return model

    def load_saved_model(self, path: str):
        if self.is_running_in_colab():
            model_ = self.load_model(f"{self.root_path}/{path}")
        elif self.is_running_in_kaggle():
            model_ = self.load_model(f"{self.root_path}/{path}")
        else:
            model_ = self.load_model(f"{path}")
        return model_

    def save_dir_path(
        self, name: str = "Especializacion_Monografia/data/models_trained"
    ):
        save_root_path = self.load_google_drive_dir()
        os.makedirs(f"{save_root_path}/{name}", exist_ok=True)
        return f"{save_root_path}/{name}"

    def get_repo_from_git(
        self, github_repo_path: str, repo_name: str, env_key="GITHUB_TOKEN"
    ):
        branch_zip = github_repo_path.split("/")[-1]
        branch_name = branch_zip.split(".")[0]
        if self.is_running_in_colab() or self.is_running_in_kaggle():
            if self.is_running_in_colab():
                from google.colab import userdata

                token = userdata.get(env_key)

            if self.is_running_in_kaggle():
                from kaggle_secrets import UserSecretsClient

                token = UserSecretsClient().get_secret(env_key)
            token_header = f"Authorization: token {token}"
            download_and_extract(
                github_repo_path,
                branch_zip,
                repo_name,
                branch_name,
                token_header,
                self.root_path,
            )
            # Path to the directory you want to add
            path_to_add = os.path.join(self.root_path, repo_name)
            # Add the path to sys.path
            if path_to_add not in sys.path:
                sys.path.append(path_to_add)


def data_selection_for_evaluation(
    model_,
    dataset,
    time_stamp,
    num_look_back_steps,
    num_forecast_steps,
    columns_for_training,
    scaler_x,
    scaler_y,
    date_index,
    target_column,
):
    initial_timestamp_ = pd.to_datetime(time_stamp)
    initial_position = dataset[dataset[date_index] == initial_timestamp_].index[0]
    sub_set = dataset[
        initial_position : (initial_position + num_look_back_steps + num_forecast_steps)
    ]
    x_test_ = scaler_x.transform(sub_set[:num_look_back_steps][columns_for_training])
    x_test_ = x_test_.reshape(-1, x_test_.shape[0], x_test_.shape[1])
    y_pred_ = model_.predict(x_test_)
    try:
        y_pred_ = scaler_y.inverse_transform(y_pred_).flatten()
    except ValueError:
        y_pred_ = scaler_y.inverse_transform(
            y_pred_.reshape(-1, y_pred_.shape[0])
        ).flatten()
    y_true_ = dataset[
        initial_position
        + num_look_back_steps : initial_position
        + num_look_back_steps
        + num_forecast_steps
    ][target_column].to_numpy()
    return y_true_, y_pred_, sub_set[[date_index, target_column]]
