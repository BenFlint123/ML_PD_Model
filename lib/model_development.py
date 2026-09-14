from abc import ABC, abstractmethod

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
    ):
        self.input_train_data = input_train_data
        self.input_test_data = input_test_data
        if input_holdout_data is not None:
            self.input_holdout_data = input_holdout_data
        self.independent_variables = independent_variables
        self.target_variable = target_variable
        self.model_parameters = model_parameters
        self.model_parameters["random_state"] = random_state

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self):
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
    ):
        super().__init__(
            input_train_data,
            input_test_data,
            independent_variables,
            target_variable,
            model_parameters,
            random_state=random_state,
            input_holdout_data=input_holdout_data,
        )


class LightGbmModelBuilder(ClassifierBuilder):
    def __init__(
        self,
        input_train_data: pd.DataFrame,
        input_test_data: pd.DataFrame,
        independent_variables: list,
        target_variable: str,
        model_parameters: dict,
        random_state: int = 42,
        input_holdout_data: pd.DataFrame | None = None,
    ):
        super().__init__(
            input_train_data,
            input_test_data,
            independent_variables,
            target_variable,
            model_parameters,
            random_state=random_state,
            input_holdout_data=input_holdout_data,
        )
        self.model = lgbm.LGBMClassifier(**self.model_parameters)

    def run(self):
        self.X_train, self.y_train, self.X_test, self.y_test = (
            self.split_data_by_independent_and_target_variables()
        )
        self.light_gbm_model = self.train_light_gbm_model()
        self.output_train_input_df, self.output_test_input_df = (
            self.predict_target_variable()
        )
        return (
            self.light_gbm_model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.output_train_input_df,
            self.output_test_input_df,
        )

    def split_data_by_independent_and_target_variables(self):
        X_train = self.input_train_data[self.independent_variables].copy()
        y_train = np.array(self.input_train_data[[self.target_variable]].copy()).ravel()

        X_test = self.input_test_data[self.independent_variables].copy()
        y_test = np.array(self.input_test_data[[self.target_variable]].copy()).ravel()
        return X_train, y_train, X_test, y_test

    def train_light_gbm_model(self):
        self.model_parameters["random_state"] = 42
        light_gbm_model = lgbm.LGBMClassifier(**self.model_parameters)
        light_gbm_model.fit(self.X_train, self.y_train)
        return light_gbm_model

    def predict_target_variable(self):
        output_train_input_df = self.input_train_data.copy()
        output_test_input_df = self.input_test_data.copy()

        output_train_input_df["predicted_default_flag"] = (
            self.light_gbm_model.predict_proba(self.X_train)[:, 1]
        )
        output_test_input_df["predicted_default_flag"] = (
            self.light_gbm_model.predict_proba(self.X_test)[:, 1]
        )
        return output_train_input_df, output_test_input_df
