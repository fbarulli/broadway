"""Histograms, boxplots, scatter matrix, correlation heatmap using Plotly."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def histogram(df: pd.DataFrame, col: str) -> go.Figure:
    return px.histogram(df, x=col, title=f"Distribution of {col}")


def boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    return px.box(df, y=col, title=f"Boxplot of {col}")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return go.Figure()
    corr = numeric.corr()
    return px.imshow(corr, text_auto=".2f", title="Correlation Heatmap")


def scatter_matrix(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return go.Figure()
    return px.scatter_matrix(df, dimensions=numeric.columns.tolist()[:6], title="Scatter Matrix")
