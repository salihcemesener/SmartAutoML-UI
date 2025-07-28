# STANDARD MODULES
import pandas as pd
from typing import Dict, List
from abc import ABC, abstractmethod


class DataPreprocessorHandler(ABC):
    """
    Abstract base class for data preprocessing components
    """

    @abstractmethod
    def display_info(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def init_parameters_for_col(
        self,
        df: pd.DataFrame,
        col: str = "",
        categorical_cols: List = [],
        missing_values: List = [],
        settings: Dict = {},
    ) -> None:
        pass

    @abstractmethod
    def apply_method(
        self, df: pd.DataFrame, col: str, method_name: str, outlier_indices: list = []
    ) -> tuple[pd.DataFrame, str]:
        pass

    @abstractmethod
    def run(
        self, df: pd.DataFrame, settings: dict, saved_configuration_file: str
    ) -> tuple[pd.DataFrame, dict]:
        pass

    def sync_column_config_list(self, settings: Dict, setting_key: str, columns: List):
        config_list = settings.get(setting_key, [])
        configured_cols = {
            next(iter(item)) for item in config_list if isinstance(item, dict)
        }
        if isinstance(columns, (pd.Series, pd.Index)):
            incoming_cols = set(columns.tolist())
        elif isinstance(columns, pd.DataFrame):
            incoming_cols = set(columns.columns)
        else:
            incoming_cols = set(columns)
        missing_cols = incoming_cols - configured_cols
        config_list.extend({col: []} for col in missing_cols)
        return config_list
