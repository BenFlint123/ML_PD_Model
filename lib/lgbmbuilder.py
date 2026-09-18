from typing import Self

import lightgbm as lgbm
import numpy as np
import pandas as pd

from lib.modelbuilder import ClassifierBuilder


class LightGbmModelBuilder(ClassifierBuilder):
    def build_model(self) -> None:
        self.model = lgbm.LGBMClassifier(**self.model_parameters)

    def run(self):
        if self.X_train is None:
            self.split_data_by_independent_and_target_variables()
        self.build_model()
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

    def train(self) -> Self:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Training data not set. Call "
                "split_data_by_independent_and_target_variables() first, or "
                "construct via from_split_data()."
            )
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")
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
