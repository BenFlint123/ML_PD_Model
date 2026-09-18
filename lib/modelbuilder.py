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
        input_holdout_data: pd.DataFrame | None = None,
        random_state: int = 42,
        model_parameters: dict | None = None,
        fit_parameters: dict | None = None,
        model: Any | None = None,
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
        # input datasets
        self.input_train_data = input_train_data
        self.input_test_data = input_test_data
        self.input_holdout_data = input_holdout_data
        # variable names
        self.independent_variables = independent_variables
        self.target_variable = target_variable
        # parameter setting
        self.model_parameters = model_parameters or {}
        self.fit_parameters = fit_parameters or {}
        self.random_state = random_state
        # These attributes get set by class methods. Initialising with None
        self.model = model
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
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        X_holdout: pd.DataFrame | None = None,
        y_holdout: np.ndarray | pd.Series | None = None,
        random_state: int = 42,
        model_parameters: dict | None = None,
        fit_parameters: dict | None = None,
        model: Any | None = None,
        trained_model: Any | None = None,
    ) -> Self:
        instance = cls.__new__(cls)
        # input datasets
        instance.X_train = X_train
        instance.y_train = np.array(y_train).ravel()
        instance.X_test = X_test
        instance.y_test = np.array(y_test).ravel()
        instance.X_holdout = X_holdout
        instance.y_holdout = np.array(y_holdout).ravel()
        # variable names
        instance.independent_variables = list(instance.X_train.columns)
        if (
            isinstance(instance.y_train, pd.Series)
            and instance.y_train.name is not None
        ):
            instance.target_variable = str(instance.y_train.name)
        else:
            instance.target_variable = "y_true"
        # parameter setting
        instance.model_parameters = model_parameters or {}
        instance.fit_parameters = fit_parameters or {}
        instance.random_state = random_state
        # These attributes get set by class methods. Initialising with None
        instance.model = model
        instance.trained_model = trained_model
        return instance

    @abstractmethod
    def build_model(self) -> None:
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

        def split_data_by_independent_and_target_variables(self) -> Self:
            self.X_train = self.input_train_data[self.independent_variables].copy()
            self.y_train = np.array(
                self.input_train_data[[self.target_variable]].copy()
            ).ravel()

            self.X_test = self.input_test_data[self.independent_variables].copy()
            self.y_test = np.array(
                self.input_test_data[[self.target_variable]].copy()
            ).ravel()

            if self.input_holdout_data is not None:
                self.X_holdout = self.input_holdout_data[
                    self.independent_variables
                ].copy()
                self.y_holdout = np.array(
                    self.input_holdout_data[[self.target_variable]].copy()
                ).ravel()
                return self

            return self
