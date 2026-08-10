import pandas as pd
import numpy as np
import inspect

from pandas.api.types import (
    is_numeric_dtype,
    is_float_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
    is_integer_dtype,
)
import matplotlib.ticker as mtick
from typing import Union, List
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

from python_validation_library_bf.utils.data_quality.base_dq import (
    convert2series,
    ColumnStatisticsCommon,
    ColumnOutliersCommon,
)
from python_validation_library_bf.utils.plotting_toolkit.plot_generator import PlotGenerator

DV_palette = [
    "#A8384F",
    "#9B4822",
    "#3E701A",
    "#2D6980",
    "#C03954",
    "#C64024",
    "#4C7C27",
    "#347893",
    "#F14E73",
    "#ED6335",
    "#5D9438",
    "#3D8EAE",
    "#E76E84",
    "#EC7046",
    "#739F56",
    "#469CBE",
]


class ColumnStatisticsRetail:
    """
    Provides methods to summarise or remove special values of dataframe column, pd.Series,
    np.array or list
    """

    @staticmethod
    def n_values(data: Union[pd.DataFrame, pd.Series, np.array, list]) -> int:
        """
        Calculate number of observations in provided data.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]): data in a form of pd.Series,
            pd.Dataframe column, list or
            np.array.

        Results:
            int: the number of observations in provided data.

        Examples:

            >>> from data_quality_irb_retail import ColumnStatisticsRetail
            >>> ColumnStatisticsRetail.n_values([1, 2, 2, 3])
            4

            >>> ColumnStatisticsRetail.n_values(pd.Series([1, 1, 1, 1]))
            4

            >>> ColumnStatisticsRetail.n_values(np.array([1, 2, 3, 4, 4]))
            5

            >>> ColumnStatisticsRetail.n_values(pd.DataFrame({'col1': [1, 2, 2, 3]}))
            4

        """
        data = convert2series(data)
        return len(data)

    @staticmethod
    def count_special_values(
        data: Union[pd.DataFrame, pd.Series, np.array, list],
        special_n: Union[pd.Series, np.array, list] = None,
        special_d: Union[pd.Series, np.array, list] = None,
        special_c: Union[pd.Series, np.array, list] = None,
    ) -> int:
        """
        Counts the occurrences of special values in a given dataset based on its data type.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]):
                The input data to analyze. It can be a pandas DataFrame, Series, NumPy array,
                or a list.
            special_n (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is numeric. Defaults to an
                empty list.
            special_d (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is of datetime type.
                Defaults to an empty list.
            special_c (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is of string type.
                Defaults to an empty list.

        Results:
            (int): The count of special values found in the input data.

        Notes:
            - The function determines the data type of the input and counts the occurrences of
            the corresponding
            special values based on the provided lists.
            - If no special values are provided for a specific data type, the function
            defaults to counting
            against an empty list, resulting in a count of zero.

        Examples:

            >>> import pandas as pd
            >>> from data_quality_irb_retail import ColumnStatisticsRetail

            >>> data = pd.Series([1, 2, 3, 4, 5])
            >>> special_n = [2, 4]
            >>> ColumnStatisticsRetail.count_special_values(data, special_n=special_n)
            np.int64(2)

            >>> data = pd.Series(["apple", "banana", "cherry", "apple"])
            >>> special_c = ["apple"]
            >>> ColumnStatisticsRetail.count_special_values(data, special_c=special_c)
            np.int64(2)

            >>> data = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"]))
            >>> special_d = [pd.Timestamp("2023-01-01")]
            >>> ColumnStatisticsRetail.count_special_values(data, special_d=special_d)
            np.int64(2)
        """

        # Set default arguments
        if special_n is None:
            special_n = []

        if special_d is None:
            special_d = []

        if special_c is None:
            special_c = []

        data = convert2series(data)

        if is_numeric_dtype(data):
            return data.isin(special_n).sum()

        elif is_datetime64_any_dtype(data):

            return data.isin(special_d).sum()

        elif is_string_dtype(data):

            return data.isin(special_c).sum()

    @staticmethod
    def remove_special_values(
        data: Union[pd.DataFrame, pd.Series, list],
        special_n: Union[pd.Series, np.array, list] = None,
        special_d: Union[pd.Series, np.array, list] = None,
        special_c: Union[pd.Series, np.array, list] = None,
    ) -> pd.Series:
        """
        Remove special values from a given dataset based on its data type.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.array, list]):
                The input data to analyze. It can be a pandas DataFrame, Series, NumPy array,
                or a list.
            special_n (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is numeric.
                Defaults to an empty list.
            special_d (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is of datetime type.
                Defaults to an empty list.
            special_c (Union[pd.Series, np.array, list]):
                A collection of special values to count if the data is of string type.
                Defaults to an empty list.

        Results:
            (pd.Series): data series with inserted np.NaN values in place of special values.

        Examples:

            >>> import pandas as pd
            >>> from data_quality_irb_retail import ColumnStatisticsRetail

            >>> data = pd.Series([1, 2, -999, 4, -999])
            >>> ColumnStatisticsRetail.remove_special_values(data, special_n=[-999])
            0    1.0
            1    2.0
            2    NaN
            3    4.0
            4    NaN
            dtype: float64

            >>> data = pd.Series(["apple", "banana", "unknown", "cherry"])
            >>> ColumnStatisticsRetail.remove_special_values(data, special_c=["unknown"])
            0    apple
            1    banana
            2    NaN
            3    cherry
            dtype: object

            >>> data = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "1900-01-01"]))
            >>> ColumnStatisticsRetail.remove_special_values(data, special_d=[pd.Timestamp("1900-...
            0    2023-01-01
            1    2023-01-02
            2    NaT
            dtype: datetime64[ns]
        """
        # Set default arguments
        if special_n is None:
            special_n = []

        if special_d is None:
            special_d = []

        if special_c is None:
            special_c = []

        data = convert2series(data)

        if is_numeric_dtype(data):

            return data.replace(special_n, np.nan)

        elif is_datetime64_any_dtype(data):

            return data.replace(special_d, np.nan)

        elif is_string_dtype(data):

            return data.replace(special_c, np.nan)


class ColumnSelectiveStatsRetail(
    ColumnStatisticsCommon, ColumnOutliersCommon, ColumnStatisticsRetail
):
    """
    Provides method that computes specified summary statistics of a dataframe column,
    pd.Series, np.array or list
    """

    @classmethod
    def calculate(
        cls,
        data: Union[pd.Series, pd.DataFrame, np.array, list],
        stats: List[str] = None,
        quantiles_list: List[float] = None,
        **kwargs,
    ):
        """
        Calculate specified statistics for the given data.

        This method computes various statistics for the provided data based on the
        list of statistical methods specified in the `stats` parameter. It validates
        the requested statistics, ensures required arguments are provided, and computes
        the results.

        Args:
            data (Union[pd.Series, pd.DataFrame, np.array, list]):
                The input data for which statistics are calculated.
            stats (List[str], optional): A list of statistical method names to compute.
            Each method must be a callable
            attribute of the class.
            quantiles_list (List[float], optional):
                A list of quantiles to compute if "quantile" is included in `stats`. Defaults to
                [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99].
            **kwargs:
                Additional keyword arguments required by the statistical methods provided in
                stats argument.

        Returns:
            dict: A dictionary where keys are the names of the computed statistics (or quantile
            labels)
                    and values are the corresponding computed values.

        Raises:
            ValueError: If any of the requested statistics in `stats` is not a callable attribute
            of the class.
            AttributeError: If any required argument for a statistical method is missing.

        Example:

            >>> from data_quality_irb_retail import ColumnSelectiveStatsRetail

            >>> data = [1, 2, 3, 4, 5]
            >>> stats = ["mean", "quantile"]
            >>> result = ColumnSelectiveStatsRetail.calculate(data=data, stats=stats, quantiles_list=...)
            >>> print(result)
            {'mean': np.float64(3.0), '25% pctl': np.float64(2.0), '50% pctl': np.float64(3.0),
            '75% pctl':
                np.float64(4.0)}
        """

        # Set default arguments
        args = kwargs.copy()
        # Initial output
        stats_dict = {}

        # Validation
        for stat in stats:
            if hasattr(cls, stat) and callable(getattr(cls, stat)):
                continue
            raise ValueError(f"'{stat}' is not acceptable")

        for stat in stats:

            method = getattr(cls, stat)
            sig = inspect.signature(method)
            required_args = list(sig.parameters)
            required_kwargs_without_defaults = [
                name
                for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
                and name not in ("cls", "data")
            ]

            for required_kwarg in required_kwargs_without_defaults:

                if required_kwarg == "q":
                    if quantiles_list is None:
                        quantiles_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

                elif required_kwarg not in kwargs:
                    raise AttributeError(f"Missing attribute '{required_kwarg}'")

            # compute method on arguments
            if stat == "quantile":
                for q in quantiles_list:
                    # format quantiles key names
                    key = f"{round(100 * q)}% pctl"
                    stats_dict[key] = method(
                        data, q, **{k: v for k, v in args.items() if k in required_args}
                    )
            else:
                stats_dict[stat] = method(
                    data, **{k: v for k, v in args.items() if k in required_args}
                )

        return stats_dict


class DataFrameSelectiveStatsRetail:
    """
    Provides a method that summarises each column of pandas DataFrame
    """

    @staticmethod
    def calculate(
        df: pd.DataFrame | None = None, stats: List[str] = None, **kwargs
    ) -> dict:
        """
        Calculate various statistics for each column in a given DataFrame.

        This method computes specified statistics for each column in the provided DataFrame.
        It supports handling special values separately and allows for the removal of these
        special values before calculating other statistics.

        Args:
            df (pd.DataFrame | None):
                The input DataFrame. If None, an empty DataFrame is used.
            stats (List[str], optional): A list of statistical method names to compute.
            Each method must be a callable
            attribute of the class.
            quantiles_list (List[float], optional):
                A list of quantiles to compute if "quantile" is included in `stats`. Defaults to
                [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99].
            **kwargs: Additional keyword arguments passed to the
            `ColumnSelectiveStatsRetail.calculate` method.

        Returns:
            dict: A dictionary where each key is a column name from the DataFrame, and the value
            is another dictionary containing the calculated statistics for that column.

        Example:

            >>> import pandas as pd
            >>> from data_quality_irb_retail import DataFrameSelectiveStatsRetail

            >>> data = {
            ...     "col1": [1, 2, 3, 4, 5],
            ...     "col2": [10, 20, 30, 40, 50]
            ... }
            >>> df = pd.DataFrame(data)
            >>> stats = ["count_special_values", "mean", "std", "quantile"]
            >>> quantiles = [0.25, 0.5, 0.75]
            >>> results = DataFrameSelectiveStatsRetail.calculate(df=df, stats=stats, quantiles_list=...)
            >>> print(results)
            {
                "col1": {
                    "count_special_values": np.int64(0),
                    "mean": np.float64(3.0),
                    "std": np.float64(1.5811388300841898),
                    "25% pctl": np.float64(2.0), "50% pctl": np.float64(3.0),
                    "75% pctl": np.float64(4.0)
                },
                "col2": {
                    "count_special_values": np.int64(0),
                    "mean": np.float64(30.0),
                    "std": np.float(15.811388300841896),
                    "25% pctl": np.float64(20.0), "50% pctl": np.float64(30.0),
                    "75% pctl": np.float64(40.0)
                }
            }
        """

        if df is None:
            df = pd.DataFrame()

        results = {}

        for col in list(df):
            series = df[col]

            results[col] = {}
            # count special values
            if "count_special_values" in stats:
                special_count = ColumnSelectiveStatsRetail.calculate(
                    series, stats=["count_special_values"], **kwargs
                )["count_special_values"]
                results[col]["count_special_values"] = special_count

            other_stats = [s for s in stats if s != "count_special_values"]
            # remove special values for computation of rest of the statistics
            no_special_series = ColumnSelectiveStatsRetail.calculate(
                series, stats=["remove_special_values"], **kwargs
            )["remove_special_values"]

            column_stats_dict = ColumnSelectiveStatsRetail.calculate(
                no_special_series, other_stats, **kwargs
            )

            # update results dictionary
            results[col].update(column_stats_dict)

        return results


def get_summary(
    dataframe: pd.DataFrame | None = None, stats: List[str] = None, **kwargs
) -> pd.DataFrame:
    """
    This function performs a data quality analysis on a given dataframe.\
    Precisely, for every column of the dataframe, it calculates
    - missing and special values
    - outliers for numerical values following the Tukey's method (records are marked as
    outliers if they lay below
    Q1 - 1.5 \\times IQR or above Q3 + 1.5 \\times IQR)
    - calculates distribution percentiles for the non-special values of each column
    (min, 1pct, 5pct, 25pct, median,
    75pct, 95pct, 99pct and max)

    Args:
        dataframe (pd.DataFrame | None, optional): The input DataFrame to analyze. If None,
        an empty DataFrame is used.
        stats (List[str], optional): A List of statistical metrics to calculate. Defaults to a
        predefined set of
        metrics: ["mean", "std", "cv", "min", "max", "mode", "missing", "quantile",
                        "count_special_values", "n_values", "outliers_iqr_count"].
        **kwargs: Additional keyword arguments passed to the
        `DataFrameSelectiveStatsRetail.calculate` method.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics. The columns are renamed
        for clarity:
            - "50% pctl" -> "median"
            - "count_special_values" -> "special"
            - "cv" -> "CV"
            - "std" -> "stdev"
            - "n_values" -> "records"
            - "outliers_iqr_count" -> "outliers"

        Additional calculated columns include:
            - "special (%)": Percentage of special values relative to the total records.
            - "missing (%)": Percentage of missing values relative to non-special records.
            - "outliers (%)": Percentage of outliers relative to non-special records.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from data_quality_irb_retail import get_summary
        >>> data = {
        ...     "A": [1, 2, 3, 4, 5, np.nan],
        ...     "B": [10, 20, 30, 40, 50, 60],
        ...     "C": ["a", "b", "a", "b", "a", "b"]
        ... }
        >>> df = pd.DataFrame(data)
        >>> stats = ["mean", "std", "missing", "quantile"]
        >>> quantiles = [0.25, 0.5, 0.75]
        >>> special_values = [1, 2]
        >>> summary = get_summary(
        ...     dataframe=df,
        ...     stats=stats,
        ...     quantiles_list=quantiles,
        ...     special_n=special_values
        ... )
        >>> print(summary)
              mean    stdev missing 25% pctl  median 75% pctl
        A      4.0  1.000000    3.0      3.5     4.0      4.5
        B     35.0  18.708287    0.0     22.5    35.0     47.5
        C      NaN       NaN    0.0      NaN     NaN      NaN
    """

    if dataframe is None:
        dataframe = pd.DataFrame()

    if not stats:
        stats = [
            "mean",
            "std",
            "cv",
            "min",
            "max",
            "mode",
            "missing",
            "quantile",
            "count_special_values",
            "n_values",
            "outliers_iqr_count",
        ]

    dict_stats = DataFrameSelectiveStatsRetail.calculate(dataframe, stats, **kwargs)
    df_results = pd.DataFrame(dict_stats)
    df_res = df_results.transpose()
    df_res.rename(
        columns={
            "50% pctl": "median",
            "count_special_values": "special",
            "cv": "CV",
            "std": "stdev",
            "n_values": "records",
            "outliers_iqr_count": "outliers",
        },
        inplace=True,
    )

    if "count_special_values" in stats and "n_values" in stats:
        df_res.loc[:, "special (%)"] = (
            df_res.loc[:, "special"] / df_res.loc[:, "records"]
        )
    if "missing" in stats and "count_special_values" in stats and "n_values" in stats:
        df_res.loc[:, "missing (%)"] = df_res.loc[:, "missing"] / (
            df_res.loc[:, "records"] - df_res.loc[:, "special"]
        )
    if (
        "outliers_iqr_count" in stats
        and "count_special_values" in stats
        and "n_values" in stats
    ):
        df_res.loc[:, "outliers (%)"] = df_res.loc[:, "outliers"] / (
            df_res.loc[:, "records"] - df_res.loc[:, "special"]
        )

    return df_res


def get_summary_with_charts(
    df: pd.DataFrame | None = None,
    special_n: Union[pd.Series, np.array, list] = None,
    special_d: Union[pd.Series, np.array, list] = None,
    special_c: Union[pd.Series, np.array, list] = None,
    grouping: str = None,
    charts: bool = False,
    max_categories: int = 10,
    palette: List[str] = DV_palette,
    max_xticks: int = 10,
    xticks_rotation: float = 90,
) -> pd.DataFrame:
    """
    Performs a data quality analysis on a given dataframe. It also can produce charts to to help
    perform the
    DQ analyses. It accepts up to one grouping criterion (e.g. Snapshot, Segment)

    Precisely, for every column of the dataframe, it calculates
    - missing and special values
    - outliers for numerical values following the Tukey's method (records are marked as outliers
    if they lay below
    Q1 - 1.5IQR, or above Q3 + 1.5IQR)
    - calculates distribution percentiles for the non-special values of each column (min, 1pct,
    5pct, 25pct, median,
    75pct, 95pct, 99pct and max)

    It additionally can plot, for every column of the dataframe and every occurrence of the
    grouping criterion
    - SMO (Special (%), Missing (%), Outliers (%)) line chart, that will show those magnitudes
    per grouping
    - For the numerical variables, a box plot without outliers per grouping
    - For the categorical variables, a stacked histogram per grouping. The function will group
    the categories with less
    occurrences than the parameter max_categories

    Args:
        dataframe (pd.DataFrame): Dataset to analyse
        quantiles_list: (List[float]): a list of probabilities for the computation of quantiles
        special_n (Union[pd.Series, np.array, list]): a series of numerical special values
        special_d (Union[pd.Series, np.array, list]): a series of datetime special values
        special_c (Union[pd.Series, np.array, list]): a series of categorical special values
        grouping (str): name of the column of the DataFrame
        charts (bool): if True, it will produce the charts
        max_categories (int): the function will leave that number of categories for the
        histogram plots, it will group the rest
            under 'Other'
        palette (List[str]): a list of colours to be used as a palette for plotting the charts
        max_xticks (int): It sets the maximum number of ticks printed in the charts
        xticks_rotation (float): It sets the rotation of the labels of the x-axis of the charts

    Returns:
        results: DataFrame / Dictionary
            If grouping = None, the function returns a DataFrame containing results of the
            analysis
            If grouping != None, the function returns a dictionary containing a DataFrame
            with results of the analysis for each occurrence of grouping. The keys of the
            dictionary are those occurrences
        charts:
            If charts = True, the function produces the following charts for each column in df
            that is not grouping
            - SMO (Special (%), Missing (%), Outliers (%)) line chart, that will show those
            magnitudes per grouping
            - For the numerical variables, a box plot without outliers per grouping
            - For the categorical variables, a stacked histogram per grouping. The function
            will group the categories
            with less occurrences than the parameter max_categories

    Examples:
        Example 1: Basic summary without charts

            >>> import pandas as pd
            >>> from data_quality_irb_retail import get_summary_with_charts
            >>> data = pd.DataFrame({
            ...     'age': [25, 30, 35, 40, 45],
            ...     'income': [50000, 60000, 70000, 80000, 90000],
            ...     'gender': ['M', 'F', 'M', 'F', 'M']
            ... })
            >>> summary = get_summary_with_charts(df=data)
            >>> print(summary)
                   special    mean       stdev ... special (%) missing (%) outliers (%)
            age          0   35.0    7.905694 ...       0.00%       0.00%        0.00%
            income       0 70000.0 15811.388301 ...     0.00%       0.00%        0.00%
            gender       0              ...       0.00%       0.00%        0.00%

        Example 2: Summary with charts

            >>> summary_with_charts = get_summary_with_charts(df=data, charts=True)
            >>> print(summary_with_charts)
                   special    mean       stdev ... special (%) missing (%) outliers (%)
            age          0   35.0    7.905694 ...       0.00%       0.00%        0.00%
            income       0 70000.0 15811.388301 ...     0.00%       0.00%        0.00%
            gender       0              ...       0.00%       0.00%        0.00%

        Example 3: Grouped summary

            >>> grouped_summary = get_summary_with_charts(df=data, grouping='gender')
            >>> print(grouped_summary)
            OrderedDict({'M':    special    mean   stdev ... special (%) missing (%) outliers (%)
            age              0   35.0   10.0 ...       0.00%       0.00%        0.00%
            income           0 70000.0 20000.0 ...    0.00%       0.00%        0.00%
            gender           0          ...       0.00%       0.00%        0.00%
            [3 rows x 20 columns], 'F':    special    mean       stdev ... special (%) missing
            age              0   35.0    7.071068 ...    0.00%       0.00%        0.00%
            income           0 70000.0 14142.135624 ...  0.00%       0.00%        0.00%
            gender           0          ...       0.00%       0.00%        0.00%
            [3 rows x 20 columns]})

        Example 4: Summary with special values

            >>> special_values = [30, 40]
            >>> summary_with_specials = get_summary_with_charts(df=data, special_n=special_values)
                   special    mean       stdev ... special (%) missing (%) outliers (%)
            age          2   35.0    10.0 ...      40.00%      66.67%        0.00%
            income       0 70000.0 15811.388301 ...  0.00%       0.00%        0.00%
            gender       0          ...       0.00%       0.00%        0.00%
            [3 rows x 20 columns]

        Example 5: Grouped summary with charts

            >>> grouped_summary_with_charts = get_summary_with_charts(
            ...     df=data, grouping='gender', charts=True, palette=['blue', 'green']
            ... )
            OrderedDict({'M':    special    mean   stdev ... special (%) missing (%) outliers (%)
            age              0   35.0   10.0 ...       0.00%       0.00%        0.00%
            income           0 70000.0 20000.0 ...    0.00%       0.00%        0.00%
            gender           0          ...       0.00%       0.00%        0.00%
            [3 rows x 20 columns], 'F':    special    mean       stdev ... special (%) missing
            age              0   35.0    7.071068 ...    0.00%       0.00%        0.00%
            income           0 70000.0 14142.135624 ...  0.00%       0.00%        0.00%
            gender           0          ...       0.00%       0.00%        0.00%
            [3 rows x 20 columns]})
    """
    special_n_2d = special_n
    special_c_2d = special_c
    special_d_2d = special_d

    if special_n is None:
        special_n = []

    if special_c is None:
        special_c = []

    if special_d is None:
        special_d = []

    if grouping is None:
        # Charts for no grouping
        if charts:

            for current_risk_driver in list(df):

                column = df[current_risk_driver]

                # for numeric variables we plot the boxplot
                if is_numeric_dtype(column):
                    # create summary df
                    print(
                        "Special values for variable ",
                        current_risk_driver,
                        " are ",
                        special_n,
                    )

                    column_wo_specials = (
                        ColumnSelectiveStatsRetail.remove_special_values(
                            column,
                            special_n=special_n,
                            special_d=special_d,
                            special_c=special_c,
                        )
                    )
                    print(f"column_wo_specials: {column_wo_specials}")
                    colors = [palette[0], "red", "blue"]
                    pg = PlotGenerator(colors)
                    imr_plotter = pg.get_builder()
                    fig_item = imr_plotter.create_plot(
                        "box",
                        plot_params=(
                            [column_wo_specials.values],
                            dict(showfliers=False, color=palette[0]),
                        ),
                        axis_params=dict(
                            ylabel=None,
                            xlabel=None,
                            title=current_risk_driver + " - Boxplot",
                        ),
                    )

                elif is_string_dtype(column):

                    print(
                        "Special values for variable ",
                        current_risk_driver,
                        " are ",
                        special_c,
                    )

                    column1 = column.copy()
                    column_wo_specials = (
                        ColumnSelectiveStatsRetail.remove_special_values(
                            column1,
                            special_n=special_n,
                            special_d=special_d,
                            special_c=special_c,
                        )
                    )
                    df_counts = column_wo_specials.value_counts().sort_values(
                        ascending=False
                    )
                    df_plot = df_counts[: min(max_categories, df_counts.shape[0])]
                    series_top = df_counts[df_counts.index.isin(df_plot.index)]
                    series_others = df_counts[~df_counts.index.isin(df_plot.index)]

                    # add a row with others count
                    series_top["Other"] = series_others.sum()
                    histogram_data = pd.Series(
                        series_top.index.repeat(series_top.values)
                    )

                    fig_item = imr_plotter.create_plot(
                        "hist",
                        plot_params=(
                            [histogram_data],
                            dict(
                                stat="percent",
                                multiple="stack",
                                color=palette[0],
                                edgecolor="black",
                            ),
                        ),
                        axis_params=dict(
                            ylabel=None,
                            xlabel=None,
                            title=current_risk_driver + " - Histogram",
                        ),
                    )
                    ax = fig_item.get_ax()
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

                    ax.tick_params(axis="x", labelrotation=xticks_rotation)

                elif is_datetime64_any_dtype(column):

                    print(
                        "Special values for variable ",
                        current_risk_driver,
                        " are ",
                        special_d,
                    )

                else:

                    print("No type recognized for variable ", current_risk_driver)

        output = get_summary(
            dataframe=df,
            special_c=special_c,
            special_d=special_d,
            special_n=special_n,
        )

        # Format the output
        output_f = output.copy()
        output_f[["special (%)", "missing (%)", "outliers (%)"]] = output_f[
            ["special (%)", "missing (%)", "outliers (%)"]
        ].applymap(lambda x: "{:,.2%}".format(x) if pd.isna(x) is False else x)
        output_f[["records", "special", "missing", "outliers"]] = output_f[
            ["records", "special", "missing", "outliers"]
        ].applymap(lambda x: "{:,.0f}".format(x) if pd.isna(x) is False else x)
        output_f[
            [
                "min",
                "1% pctl",
                "5% pctl",
                "25% pctl",
                "median",
                "75% pctl",
                "95% pctl",
                "99% pctl",
                "max",
                "stdev",
                "CV",
            ]
        ] = output_f[
            [
                "min",
                "1% pctl",
                "5% pctl",
                "25% pctl",
                "median",
                "75% pctl",
                "95% pctl",
                "99% pctl",
                "max",
                "stdev",
                "CV",
            ]
        ].applymap(
            lambda x: (
                "{:,.2f}".format(x)
                if (is_numeric_dtype(x) and pd.isna(x) is False)
                else x
            )
        )
        output_f[
            [
                "min",
                "1% pctl",
                "5% pctl",
                "25% pctl",
                "median",
                "75% pctl",
                "95% pctl",
                "99% pctl",
                "max",
                "stdev",
                "CV",
            ]
        ] = output_f[
            [
                "min",
                "1% pctl",
                "5% pctl",
                "25% pctl",
                "median",
                "75% pctl",
                "95% pctl",
                "99% pctl",
                "max",
                "stdev",
                "CV",
            ]
        ].applymap(
            lambda x: (
                pd.to_datetime(str(x)).strftime("%b-%y")
                if (is_datetime64_any_dtype(x) and pd.isna(x) is False)
                else x
            )
        )
        output_f = output_f.applymap(
            lambda x: " " if (type(x) is float and pd.isna(x)) else x
        )

        return output_f

    else:

        results = OrderedDict()
        results_f = OrderedDict()  # Formatted output

        group_set = list(set(df[grouping]))
        if is_numeric_dtype(df[grouping]) | is_datetime64_any_dtype(df[grouping]):
            group_set.sort()

        for i in range(len(group_set)):
            df_i = df.loc[df[grouping] == group_set[i]]
            dq_special2d_output = get_summary(
                df_i,
                special_n=special_n_2d,
                special_c=special_c_2d,
                special_d=special_d_2d,
            )
            results.update({group_set[i]: dq_special2d_output})

            # Format the output
            dq_special2d_output_f = dq_special2d_output.copy()
            dq_special2d_output_f[["special (%)", "missing (%)", "outliers (%)"]] = (
                dq_special2d_output_f[
                    ["special (%)", "missing (%)", "outliers (%)"]
                ].applymap(lambda x: "{:,.2%}".format(x) if pd.isna(x) is False else x)
            )
            dq_special2d_output_f[["records", "special", "missing", "outliers"]] = (
                dq_special2d_output_f[
                    ["records", "special", "missing", "outliers"]
                ].applymap(lambda x: "{:,.0f}".format(x) if pd.isna(x) is False else x)
            )
            dq_special2d_output_f[
                [
                    "min",
                    "1% pctl",
                    "5% pctl",
                    "25% pctl",
                    "median",
                    "75% pctl",
                    "95% pctl",
                    "99% pctl",
                    "max",
                    "stdev",
                    "CV",
                ]
            ] = dq_special2d_output_f[
                [
                    "min",
                    "1% pctl",
                    "5% pctl",
                    "25% pctl",
                    "median",
                    "75% pctl",
                    "95% pctl",
                    "99% pctl",
                    "max",
                    "stdev",
                    "CV",
                ]
            ].applymap(
                lambda x: (
                    "{:,.2f}".format(x)
                    if (is_numeric_dtype(x) and pd.isna(x) is False)
                    else x
                )
            )
            dq_special2d_output_f[
                [
                    "min",
                    "1% pctl",
                    "5% pctl",
                    "25% pctl",
                    "median",
                    "75% pctl",
                    "95% pctl",
                    "99% pctl",
                    "max",
                    "stdev",
                    "CV",
                ]
            ] = dq_special2d_output_f[
                [
                    "min",
                    "1% pctl",
                    "5% pctl",
                    "25% pctl",
                    "median",
                    "75% pctl",
                    "95% pctl",
                    "99% pctl",
                    "max",
                    "stdev",
                    "CV",
                ]
            ].applymap(
                lambda x: (
                    pd.to_datetime(str(x)).strftime("%b-%y")
                    if (is_datetime64_any_dtype(x) and pd.isna(x) is False)
                    else x
                )
            )
            dq_special2d_output_f = dq_special2d_output_f.applymap(
                lambda x: " " if (type(x) is float and pd.isna(x)) else x
            )

            results_f.update({group_set[i]: dq_special2d_output_f})

        if charts is True:

            smo_df = results.get(list(results.keys())[0])[
                ["special (%)", "missing (%)", "outliers (%)"]
            ]
            colors = [palette[0], "red", "blue"]

            pg = PlotGenerator(colors)
            imr_plotter = pg.get_builder()

            for j in range(smo_df.shape[0]):
                current_risk_driver = smo_df.iloc[[j]].index.values[0]

                if current_risk_driver != grouping:
                    smo_var_df_chart = pd.DataFrame(
                        columns=["special (%)", "missing (%)", "outliers (%)", grouping]
                    )

                    for i in range(len(results.keys())):
                        smo_df = results.get(list(results.keys())[i])[
                            ["special (%)", "missing (%)", "outliers (%)"]
                        ]
                        smo_var_df = smo_df.iloc[[j]].copy()
                        smo_var_df[grouping] = list(results.keys())[i]
                        smo_var_df_chart = pd.concat([smo_var_df_chart, smo_var_df])

                    chart_df_smo = (
                        smo_var_df_chart.loc[current_risk_driver]
                        .reset_index()
                        .melt(
                            id_vars=grouping,
                            value_vars=["special (%)", "missing (%)", "outliers (%)"],
                        )
                    )

                    if is_string_dtype(df[current_risk_driver]):
                        chart_df_smo = chart_df_smo.loc[
                            lambda chart_df_smo: chart_df_smo["variable"]
                            != "outliers (%)",
                            :,
                        ]

                        palette_smo = palette[:2]
                    else:
                        palette_smo = palette[:3]

                    if is_datetime64_any_dtype(df[grouping]):
                        chart_df_smo = chart_df_smo.sort_values(by=grouping)
                        chart_df_smo[grouping + "_f"] = chart_df_smo[
                            grouping
                        ].dt.strftime("%b-%y")
                    else:
                        chart_df_smo = chart_df_smo.sort_values(by=grouping)
                        chart_df_smo[grouping + "_f"] = chart_df_smo[grouping].astype(
                            "string"
                        )
                    # print(chart_df_smo)

                    with sns.axes_style("white"):
                        fig_item = imr_plotter.create_figure(
                            "multiple subplots",
                            1,
                            2,
                            figure_params=dict(figsize=(10, 4)),
                        )

                    ax1 = fig_item.create_plot(
                        "line",
                        plot_params=(
                            [chart_df_smo],
                            dict(
                                x=grouping + "_f",
                                y="value",
                                hue="variable",
                                palette=palette_smo,
                            ),
                        ),
                        axis_params=dict(
                            ylabel=None,
                            xlabel=None,
                            title=current_risk_driver + " - Data issues",
                        ),
                        index=0,
                    )
                    ax1.tick_params(axis="x", labelrotation=xticks_rotation)

                    ax = fig_item.get_ax()
                    ax.get_legend().set_title(None)
                    ax.legend().set_title("")
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
                    if len(ax.get_xticks()) > (max_xticks + 2):
                        ax.xaxis.set_major_locator(
                            plt.LinearLocator(numticks=(max_xticks + 2))
                        )
                    xticks_smo = ax.get_xticks()
                    xlabels_smo = ax.get_xticklabels()

                    if is_numeric_dtype(df[current_risk_driver]):
                        print(
                            "Special values for variable ",
                            current_risk_driver,
                            " are ",
                            special_n,
                        )

                        if is_datetime64_any_dtype(df[grouping]):
                            chart_df_box = df[[grouping, current_risk_driver]].copy()
                            chart_df_box[grouping + "_f"] = chart_df_box[
                                grouping
                            ].dt.strftime("%b-%y")
                        else:
                            chart_df_box = df[[grouping, current_risk_driver]].copy()
                            chart_df_box[grouping + "_f"] = chart_df_box[grouping]

                        chart_df_box = chart_df_box.loc[
                            ~chart_df_box[current_risk_driver].isin(special_n), :
                        ]
                        chart_df_box = chart_df_box.sort_values(by=grouping)

                        with sns.axes_style("white"):

                            ax2 = fig_item.create_plot(
                                "box",
                                plot_params=(
                                    [chart_df_box],
                                    dict(
                                        x=grouping + "_f",
                                        y=current_risk_driver,
                                        showfliers=False,
                                        palette=palette,
                                    ),
                                ),
                                axis_params=dict(
                                    ylabel=None,
                                    xlabel=None,
                                    title=current_risk_driver + " - Boxplot",
                                ),
                                index=1,
                            )

                        ax2.tick_params(axis="x", labelrotation=xticks_rotation)

                        ax = fig_item.get_ax()
                        ax.set_xticks(
                            xticks_smo
                        )  # Apply ticks consistent with the SMO chart
                        ax.set_xticklabels(
                            xlabels_smo
                        )  # Apply labels consistent with the SMO chart

                    elif is_string_dtype(df[current_risk_driver]):

                        print(
                            "Special values for variable ",
                            current_risk_driver,
                            " are ",
                            special_c,
                        )
                        chart_df_hist = df[[grouping, current_risk_driver]].copy()
                        chart_df_hist = chart_df_hist.loc[
                            ~chart_df_hist[current_risk_driver].isin(special_c), :
                        ]
                        chart_df_hist = chart_df_hist.sort_values(by=grouping)
                        if is_datetime64_any_dtype(chart_df_hist[grouping]):
                            chart_df_hist[grouping + "_f"] = chart_df_hist[
                                grouping
                            ].dt.strftime("%b-%y")
                        else:
                            chart_df_hist[grouping + "_f"] = chart_df_hist[
                                grouping
                            ].astype("string")
                        chart_df_hist_aux = (
                            chart_df_hist.groupby([current_risk_driver])
                            .size()
                            .reset_index()
                        )
                        chart_df_hist_aux = chart_df_hist_aux.sort_values(
                            by=0, ascending=False
                        ).reset_index(drop=True)
                        chart_df_hist_aux[current_risk_driver + "_f"] = "Other"
                        chart_df_hist_aux.iloc[
                            chart_df_hist_aux.index[
                                range(min(chart_df_hist_aux.shape[0], max_categories))
                            ],
                            chart_df_hist_aux.columns.get_loc(
                                current_risk_driver + "_f"
                            ),
                        ] = chart_df_hist_aux.iloc[
                            chart_df_hist_aux.index[
                                range(min(chart_df_hist_aux.shape[0], max_categories))
                            ],
                            chart_df_hist_aux.columns.get_loc(current_risk_driver),
                        ]

                        # we update chart_df_hist to have the formatted grouping and
                        # current_risk_driver

                        chart_df_hist = chart_df_hist.merge(
                            chart_df_hist_aux, on=current_risk_driver, how="left"
                        )
                        chart_df_hist = chart_df_hist.drop(
                            [current_risk_driver, grouping], axis=1
                        )
                        chart_df_hist = chart_df_hist.rename(
                            columns={
                                current_risk_driver + "_f": current_risk_driver,
                                grouping + "_f": grouping,
                            }
                        )
                        chart_df_hist = chart_df_hist.rename(
                            columns={current_risk_driver + "_f": current_risk_driver}
                        )

                        with sns.axes_style("white"):

                            ax3 = fig_item.create_plot(
                                "hist",
                                plot_params=(
                                    [chart_df_hist],
                                    dict(
                                        x=grouping,
                                        hue=current_risk_driver,
                                        stat="count",
                                        multiple="stack",
                                        palette=palette,
                                    ),
                                ),
                                axis_params=dict(
                                    ylabel=None,
                                    xlabel=None,
                                    title=current_risk_driver + " - Histogram",
                                ),
                                index=1,
                            )

                        ax3.tick_params(axis="x", labelrotation=xticks_rotation)

                        ax = fig_item.get_ax()
                        ax.set_xticks(
                            xticks_smo
                        )  # Apply ticks consistent with the SMO chart
                        ax.set_xticklabels(
                            xlabels_smo
                        )  # Apply labels consistent with the SMO

                    elif is_datetime64_any_dtype(df[current_risk_driver]):

                        print(
                            "Special values for variable ",
                            current_risk_driver,
                            " are ",
                            special_d,
                        )

                    else:

                        print("No type recognized for variable ", current_risk_driver)

        return results_f
