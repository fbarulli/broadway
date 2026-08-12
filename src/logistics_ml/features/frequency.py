import pandas as pd


def make_frequency_encoding(
    train: pd.DataFrame,
    cols: list[str],
    feature_name: str,
) -> pd.DataFrame:
    """
    Compute frequency encoding from the training data.
    """

    stats = (
        train.groupby(cols)
        .size()
        .rename(feature_name)
        .reset_index()
    )

    return stats


def apply_frequency_encoding(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    cols: list[str],
    feature_name: str,
    fill_value: int = 0,
) -> pd.DataFrame:
    """
    Merge a frequency encoding into a dataframe.
    """

    df = df.merge(
        stats,
        on=cols,
        how="left",
    )

    df[feature_name] = (
        df[feature_name]
        .fillna(fill_value)
        .astype("int32")
    )

    return df
