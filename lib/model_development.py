from abc import ABC, abstractmethod
from typing import Any, Self

import lightgbm as lgbm
import numpy as np
import pandas as pd


class ModelBuilder(ABC):
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
        self._build_model()

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


class LightGbmModelBuilder(ClassifierBuilder):
    def _build_model(self) -> None:
        self.model = lgbm.LGBMClassifier(**self.model_parameters)

    def run(self):
        if self.X_train is None:
            self.split_data_by_independent_and_target_variables()
        self.train()
        self.predict_train_set()
        self.predict_test_set()
        return_dict = {
            "model": self.trained_model,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "pred_train": self.train_predicted,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "pred_test": self.test_predicted,
        }
        if self.X_holdout is not None:
            self.predict_holdout_set()
            return_dict["X_holdout"] = self.X_holdout
            return_dict["y_holdout"] = self.y_holdout
            return_dict["pred_holdout"] = self.holdout_predicted

        return return_dict

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
            self.X_holdout = self.input_holdout_data[self.independent_variables].copy()
            self.y_holdout = np.array(
                self.input_holdout_data[[self.target_variable]].copy()
            ).ravel()
            return self

        return self

    def train(self) -> Self:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Training data not set. Call "
                "split_data_by_independent_and_target_variables() first, or "
                "construct via from_split_data()."
            )
        self.trained_model = self.model.fit(self.X_train, self.y_train)
        return self

    def predict_train_set(self) -> Self:
        if self.trained_model is None:
            raise RuntimeError(
                "Model not trained. Run 'train' before any 'predict' methods."
            )
        assert self.X_train is not None  # guaranteed by train()

        self.train_predicted = np.asarray(
            self.trained_model.predict_proba(self.X_train)
        )[:, 1]

        return self

    def predict_test_set(self) -> Self:
        if self.trained_model is None:
            raise RuntimeError(
                "Model not trained. Run 'train' before any 'predict' methods."
            )
        assert self.X_test is not None  # guaranteed by train()

        self.test_predicted = np.asarray(self.trained_model.predict_proba(self.X_test))[
            :, 1
        ]

        return self

    def predict_holdout_set(self) -> Self:
        if self.trained_model is None:
            raise RuntimeError(
                "Model not trained. Run 'train' before any 'predict' methods."
            )
        if self.X_holdout is None:
            raise RuntimeError(
                "Holdout data not provided. Cannot predict for holdout sample."
            )

        self.holdout_predicted = np.asarray(
            self.trained_model.predict_proba(self.X_holdout)
        )[:, 1]

        return self


def main():
    from config import DATA

    dataset_path = DATA / "processed" / "basic_data_prep"
    datasets = {f.stem: pd.read_csv(f) for f in sorted(dataset_path.glob("*.csv"))}
    for k, v in datasets.items():
        print(f"{k}: \n{v}\n")

    model_params_dict = {"verbose": 2}
    results_dict = lgbm_model = LightGbmModelBuilder.from_split_data(
        datasets["X_train"],
        datasets["y_train"],
        datasets["X_test"],
        datasets["y_test"],
        model_params_dict,
    ).run()


if __name__ == "__main__":
    main()
