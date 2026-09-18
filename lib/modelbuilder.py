"""This module defines abstract base classes to build statistical models. The intention
is that the base classes provide a single contract for implementations of commonly used
models from sklearn, pytorch, stats_models, etc. for a variety of classification and
regression tasks.

Author: Ben Flint
"""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd


class ModelBuilder(ABC):
    """This class defines the minimum attribute and method set required to build and run all models"""

    def __init__(
        self,
        input_train_data: pd.DataFrame,
        input_test_data: pd.DataFrame,
        independent_variables: list,
        target_variable: str,
        model_parameters: dict,
        random_state: int = 42,
        input_holdout_data: pd.DataFrame | None = None,
        trained_model: Any | None = None,
    ):
        """_summary_

        Args:
            input_train_data (pd.DataFrame): _description_
            input_test_data (pd.DataFrame): _description_
            independent_variables (list): _description_
            target_variable (str): _description_
            model_parameters (dict): _description_
            random_state (int, optional): _description_. Defaults to 42.
            input_holdout_data (pd.DataFrame | None, optional): _description_. Defaults to None.
            trained_model (Any | None, optional): _description_. Defaults to None.
        """
        self.input_train_data = input_train_data
        self.input_test_data = input_test_data
        self.input_holdout_data = input_holdout_data
        self.independent_variables = independent_variables
        self.target_variable = target_variable
        self.model_parameters = model_parameters
        self.model_parameters["random_state"] = random_state
        self.trained_model = trained_model
        self.X_train: pd.DataFrame | None = None
        self.y_train: np.ndarray | pd.Series | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: np.ndarray | pd.Series | None = None
        self.X_holdout: pd.DataFrame | None = None
        self.y_holdout: np.ndarray | pd.Series | None = None

    @classmethod
    def from_split_data(
        cls,
        X_train: pd.DataFrame | None,
        y_train: np.ndarray | pd.Series | None,
        X_test: pd.DataFrame | None,
        y_test: np.ndarray | pd.Series | None,
        model_parameters: dict,
        random_state: int = 42,
        X_holdout: pd.DataFrame | None = None,
        y_holdout: np.ndarray | pd.Series | None = None,
    ) -> Self:
        instance = cls.__new__(cls)
        instance.model_parameters = model_parameters
        instance.model_parameters["random_state"] = random_state
        instance.trained_model = None
        instance.X_train = X_train
        instance.y_train = np.array(y_train).ravel()
        instance.X_test = X_test
        instance.y_test = np.array(y_test).ravel()
        instance.X_holdout = X_holdout
        instance.y_holdout = np.array(y_holdout).ravel()
        instance._build_model()
        return instance

    @abstractmethod
    def _build_model(self) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict_train_set(self):
        pass

    @abstractmethod
    def predict_test_set(self):
        pass


class ClassifierBuilder(ModelBuilder, ABC):
    def __init__(
        self,
        input_train_data: pd.DataFrame,
        input_test_data: pd.DataFrame,
        independent_variables: list,
        target_variable: str,
        model_parameters: dict,
        random_state: int = 42,
        input_holdout_data: pd.DataFrame | None = None,
        trained_model: Any | None = None,
    ):
        super().__init__(
            input_train_data,
            input_test_data,
            independent_variables,
            target_variable,
            model_parameters,
            random_state=random_state,
            input_holdout_data=input_holdout_data,
            trained_model=trained_model,
        )
