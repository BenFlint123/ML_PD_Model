import pandas as pd
import numpy as np
from pandas.api.types import (
    is_numeric_dtype,
    is_float_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
    is_integer_dtype,
)
from typing import Union, List, Any
from scipy.stats import zscore, t


def convert2series(data: Union[pd.Series, pd.DataFrame, np.ndarray, list]) -> pd.Series:
    """
    Convert input into pandas Series.
    Parameters:
        data (pd.Series, pd.DataFrame, np.array or list): The values which are to be converted
        into pd.Series.

    Returns:
        pd.Series: values from data input converted into pandas Series type.

    Raises:
        ValueError: If data is pd.DataFrame object and has more than one column.
        TypeError: If data type is different than pd.Series, pd.DataFrame, np.array or list.

    """

    # Change input type to pd.Series
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.squeeze()
        else:
            raise ValueError(
                f"Incorrect number of columns in a dataframe. Expected 1 column, Received: "
                f"{data.shape[1]}"
            )

    elif isinstance(data, (pd.Series, np.ndarray, list, pd.DatetimeIndex)):
        data = pd.Series(data)

    else:
        raise TypeError(
            f"data should be one type of [pd.Series, pd.DataFrame, np.array, list]. Received: "
            f"{data.__class__.__name__}"
        )

    return data


class ColumnStatisticsCommon:
    """
    Provides methods useful for computation of basic statistics of a given pd.Series,
    pd.Dataframe column or
    np.array.
    """

    @staticmethod
    def mean(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate mean value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column or
            np.array.

        Results:
            (float): mean value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.mean()

        elif is_datetime64_any_dtype(data):
            return data.mean()

        elif is_string_dtype(data):
            return np.nan
        
        else: 
            return np.nan

    @staticmethod
    def std(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, list], ddof: int = 1
    ) -> float:
        """
        Calculate standard deviation value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): a sequence of data points
            ddof (int): Delta Degrees of freedom. The divisor used in the calculation
            is N - ddof, where N is the number of elements. By default, ddof=0 calculates
            the population standard deviation. Set ddof=1 for an unbiased estimate of
            the sample standard deviation.

        Results:
            (float): standard deviation value of the input data.
        Raises:
            ValueError: if ddof is different than 0 or 1.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):

            if ddof in [0, 1]:

                return data.std(ddof=ddof)

            raise ValueError(
                "Argument ddof must be either 0 (population) or 1 (sample)"
            )

            return data.std()

        elif is_datetime64_any_dtype(data):
            return np.nan

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def median(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate median value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column or
            np.array.

        Results:
            (float): median value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.median()

        elif is_datetime64_any_dtype(data):
            return data.median()

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def quantile(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, list], q: float
    ) -> float:
        """
        Calculate quantile value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            np.array or list.
            q (float): quantile to compute, which must be between 0 and 1 (inclusive).
        Results:
            (float): quantile value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.quantile(q)

        elif is_datetime64_any_dtype(data):
            return data.quantile(q)

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def min(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate min value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.

        Results:
            (float): min value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.min()

        elif is_datetime64_any_dtype(data):
            return data.min()

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def max(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate max value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column, list or
            np.array.

        Results:
            (float): max value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.max()

        elif is_datetime64_any_dtype(data):
            return data.max()

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def mode(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate mode of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column or
            np.array.

        Results:
            (float): the most frequently occuring value in the data. If multiple values have the
            same highest frequency
            the first one encountered is returned.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):

            mode = data.mode()
            if not mode.empty:
                return mode[0]
            else:
                return np.nan

        elif is_datetime64_any_dtype(data):

            mode = data.mode()
            if not mode.empty:
                return mode[0]
            else:
                return np.nan

        elif is_string_dtype(data):

            mode = data.mode()
            if not mode.empty:
                return mode[0]
            else:
                return np.nan
        else:
            return np.nan

    @staticmethod
    def cv(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> float:
        """
        Calculate Coefficient of Variation value of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.

        Results:
            (float): Coefficient of Variation value of provided data.
        """
        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.std() / data.mean()

        elif is_datetime64_any_dtype(data):
            return np.nan

        elif is_string_dtype(data):
            return np.nan
        else:
            return np.nan

    @staticmethod
    def n_categories(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> int:
        """
        Calculate number of unique values of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.

        Results:
            (int): number of unique values of provided data if data type is categorical else
            np.nan is returned.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            return data.nunique()

        else:

            return np.nan

    @staticmethod
    def unique_categories(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> str:
        """
        Return unique values of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.

        Results:
            (str): string containing a list of unique values of provided data if data type is
            categorical
            else np.nan is returned.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            value_counts = data.value_counts(normalize=True)
            return str(value_counts.index.tolist())

        else:
            return np.nan

    @staticmethod
    def category_frequencies(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    ) -> str:
        """
        Return frequencies of unique values of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.

        Results:
            (str): string containing a list of unique values and its frequencies of provided data
            if data type is categorical
            else np.nan is returned.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            value_counts = data.value_counts(normalize=True)
            return ", ".join([f"{k}: {v:.0%}" for k, v in value_counts.items()])

        else:
            return np.nan

    @staticmethod
    def rare_category(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
        rare_threshold: float = 0.01,
    ) -> str:
        """
        Return unique values of provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            rare_threshold (float): threshold that defines whether a frequency is rare or not.
        Results:
            (str): string containing a list of rare unique values of provided data if data type
            is categorical
            else np.nan is returned.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            value_counts = data.value_counts(normalize=True)
            return str(value_counts[value_counts < rare_threshold].index.tolist())
        else:
            return np.nan

    @staticmethod
    def missing_categories(data: Union[pd.DataFrame, pd.Series, list]) -> List[Any]:
        """
        Return values associated with missing categories inside data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            rare.threshold (float): threshold that defines whether a frequency is rare or not.
        Results:
            (str): string containing a list of rare unique values of provided data if data type
            is categorical
            else np.nan is returned.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            value_counts = data.value_counts(normalize=True)
            unique_categories = value_counts.index.tolist()
            return [
                cat
                for cat in unique_categories
                if pd.isna(cat)
                or str(cat).lower() in ["unknown", "na", "n/a", "missing", "mi"]
            ]

        else:
            return np.nan

    @staticmethod
    def count(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> int:
        """
        Count number of values except NaN values in data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
        Results:
            (int): count of all not-NaN values in the input data.
        """
        data = convert2series(data)

        return data.count()

    @staticmethod
    def most_frequent(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    ) -> Union[int, str]:
        """
        Find the most common value in the data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
        Results:
            (int | str): most frequent value in the input data or np.nan.
        """
        data = convert2series(data)

        if is_string_dtype(data) | is_integer_dtype(data):

            value_counts = data.value_counts(normalize=True)
            return value_counts.idxmax()

        else:
            return np.nan

    @staticmethod
    def missing(data: Union[pd.DataFrame, np.ndarray, pd.Series, list]) -> int:
        """
        Count number of missing values in the data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
        Results:
            (int): number of missing values in data.
        """
        data = convert2series(data)

        return data.isnull().sum()

    @staticmethod
    def n_values(data: Union[pd.DataFrame, pd.Series, np.ndarray, list]) -> int:
        """
        Count number of all values in the data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
        Results:
            (int): number of all values in the data.
        """
        data = convert2series(data)

        return len(data)


class ColumnOutliersCommon:
    """
    Provides useful methods for computing outliers of a given pd.Series, pd.Dataframe column,
    list or
    np.array.
    """

    z_score_threshold: float = 3
    alpha: float = 0.05
    iqr_multiplier: float = 1.5

    @classmethod
    def outliers_iqr(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        iqr_multiplier: float = None,
    ) -> pd.Series:
        """
        Detect outliers of the data based on Interquartile Range (IQR) method.

        An outlier is defined as  value that lies below Q1 - iqr_multiplier * IQR or above Q3 -
        iqr_multiplier * IQR,
        where:
        - Q1 - is the 25th percentile of the data
        - Q3 - is the 75th percentile of the data
        - IQR = Q3 - Q1.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            iqr_multiplier (float): a multiplier that determines how far from the IQR bounds a
            value must be to be
            considered as an outlier.
        Results:
            (pd.Series): A boolean Series indicating outliers (True for outliers).
        """
        iqr_multiplier = iqr_multiplier or cls.iqr_multiplier

        data = convert2series(data)

        if is_numeric_dtype(data):

            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            return (data < lower_bound) | (data > upper_bound)

        else:
            return pd.Series([], dtype="float")

    @classmethod
    def outliers_zscore(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        z_score_threshold: float = None,
    ) -> pd.Series:
        """
        Detect outliers of the data using the z-score method.

        A data point is considered an outlier if its z-score is greater than 'z_score_threshold'
        or less than
        '- z_score_threshold'. The z-score is calculated as (x-mean) / standard_deviation.


        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            
            pd.Dataframe column,
            list or np.array.
            z_score_threshold (float): The Z-score threshold above which a value is considered
            as an outlier.
        Results:
            (pd.Series): A boolean Series indicating outliers (True for outliers).
        """

        z_score_threshold = z_score_threshold or cls.z_score_threshold
        data = convert2series(data)

        if is_numeric_dtype(data):
            z_score = zscore(data.dropna())
            return np.abs(z_score) > z_score_threshold

        else:

            return pd.Series([], dtype="float")

    @classmethod
    def outliers_grubbs(
        cls, data: Union[pd.Series, pd.DataFrame, np.ndarray, list], alpha: float = None
    ) -> pd.Series:
        """
        Detect outliers of the data using Grubbs test.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            alpha (float): Significance level for the Grubbs test.
        Results:
            (pd.Series): A boolean Series indicating outliers (True for outliers).
        """

        alpha = alpha or cls.alpha
        data = convert2series(data)

        if is_numeric_dtype(data) | is_datetime64_any_dtype(data):

            n = len(data.dropna())

            if n < 3:
                return pd.Series(
                    [False] * len(data)
                )  # Grubbs' test requires at least 3 observations

            mean = data.mean()
            std_dev = data.std()

            t_critical = t.ppf(
                1 - alpha / (2 * n), n - 2
            )  # Critical value for Grubbs' test
            g_critical = ((n - 1) * t_critical) / np.sqrt(n * ((n - 2) + t_critical**2))

            return np.abs(data - mean) > g_critical * std_dev

        else:

            return pd.Series([], dtype="float")

    @classmethod
    def outliers_grubbs_pct(
        cls, data: Union[pd.Series, pd.DataFrame, np.ndarray, list], alpha: float = None
    ) -> float:
        """
        Compute frequency of outliers in the data using Grubbs method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            alpha (float): Significance level for the Grubbs test.
        Results:
            (float): frequency of outliers in the data.
        """

        return cls.outliers_grubbs(data, alpha).mean()

    @classmethod
    def outliers_zscore_pct(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        z_score_threshold: float = None,
    ) -> float:
        """
        Compute frequency of outliers in the data using z-score method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            z_score_threshold (float): The Z-score threshold above which a value is considered as
            an outlier.
        Results:
            (float): frequency of outliers in the data.
        """

        return cls.outliers_zscore(data, z_score_threshold).mean()

    @classmethod
    def outliers_iqr_pct(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        iqr_multiplier: float = None,
    ) -> float:
        """
        Compute frequency of outliers in the data using Interquartile Range method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            iqr_multiplier (float): a multiplier that determines how far from the IQR bounds a
            value must be to be
            considered as an outlier.
        Results:
            (float): frequency of outliers in the data.
        """

        return cls.outliers_grubbs(data, iqr_multiplier).mean()

    @classmethod
    def outliers_grubbs_count(
        cls, data: Union[pd.Series, pd.DataFrame, np.ndarray, list], alpha: float = None
    ) -> int:
        """
        Count number of outliers in the data using Grubbs method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            alpha (float): Significance level for the Grubbs test.
        Results:
            (int): number of outliers in the data.
        """

        return cls.outliers_grubbs(data, alpha).sum()

    @classmethod
    def outliers_zscore_count(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        z_score_threshold: float = None,
    ) -> int:
        """
        Count number of outliers in the data using z-score method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            z_score_threshold (float): The Z-score threshold above which a value is considered as
            an outlier.
        Results:
            (int): number of outliers in the data.
        """

        return cls.outliers_zscore(data, z_score_threshold).sum()

    @classmethod
    def outliers_iqr_count(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        iqr_multiplier: float = None,
    ) -> int:
        """
        Count number of outliers in the data using Interquartile Range method.
        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column,
            list or np.array.
            iqr_multiplier (float): a multiplier that determines how far from the IQR bounds a
            value must be to be
            considered as an outlier.
        Results:
            (int): number of outliers in the data.
        """

        return cls.outliers_grubbs(data, iqr_multiplier).sum()


class ColumnSelectiveStatsCommon(ColumnStatisticsCommon, ColumnOutliersCommon):
    """
    Provides method to compute chosen statistics on one-dimensional input data.
    """

    @classmethod
    def calculate(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
        stats: List[str] = None,
        ddof: int = 1,
        quantiles_list: List[float] = None,
        rare_threshold: float = 0.01,
    ):
        """
        Compute chosen statistics or outliers on the input data.
        Args:
            data: (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.DataFrame column,
            stats: (List[str]): a list containing names of all statistics to be computed
            ddof: (int): Delta Degrees of freedom. The divisor used in the calculation of s
            tandard deviation.
            quantiles_list: (List[float]): a list of probabilities for the computation of
            quantiles
            rare_threshold: (float): threshold that defines whether a frequency is rare or not.

        Returns:
            (dict): a dictionary consisting of statistic_names as keys and its corresponding
            values.
        """
        # Set default arguments
        if quantiles_list is None:
            quantiles_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

        # Initial output
        stats_dict = {}

        # Validation
        for stat in stats:
            if hasattr(cls, stat) and callable(getattr(cls, stat)):
                continue
            raise ValueError(f"'{stat}' is not acceptable")

        for stat in stats:
            if stat == "quantile":
                for q in quantiles_list:
                    method = getattr(cls, stat)
                    # format key name
                    key = f"{round(100 * q)}% pctl"
                    stats_dict[key] = method(data, q)

            elif stat == "rare_category":
                method = getattr(cls, stat)
                stats_dict[stat] = method(data, rare_threshold)

            elif stat == "std":
                method = getattr(cls, stat)
                stats_dict[stat] = method(data, ddof)

            else:
                method = getattr(cls, stat)
                stats_dict[stat] = method(data)

        return stats_dict


class DataFrameSelectiveStatsCommon:
    """
    Provides method to compute chosen statistics on a dataframe.
    """

    @staticmethod
    def calculate(
        df: pd.DataFrame = pd.DataFrame([]),
        stats: List[str] = None,
        quantiles_list: List[float] = None,
        rare_threshold: float = 0.01,
        ddof: int = 1,
    ):
        """
        Compute chosen statistics or outliers on the input dataframe.
        Args:
            df: (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.DataFrame column,
            stats: (List[str]): a list containing names of all statistics to be computed
            quantiles_list: (List[float]): a list of probabilities for the computation of
            quantiles
            rare_threshold: (float): threshold that defines whether a frequency is rare or not
            ddof: (int): Delta Degrees of freedom. The divisor used in the calculation of
            standard deviation.

        Returns:
            (dict): a dictionary consisting of column names as keys and dictionaries with
            statistic_names as keys and
            its corresponding values as a value
        """
        results = {}

        if not quantiles_list:
            quantiles_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

        for col in list(df):
            series = df[col]

            results[col] = {}

            column_stats_dict = ColumnSelectiveStatsCommon.calculate(
                series,
                stats,
                quantiles_list=quantiles_list,
                rare_threshold=rare_threshold,
                ddof=ddof,
            )

            # update results dictionary
            results[col].update(column_stats_dict)

        return results
