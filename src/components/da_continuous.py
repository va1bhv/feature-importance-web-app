from dash import Dash, dcc, html, dash_table
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import datetime
import io
import pandas as pd
from dataclasses import asdict
from pandas.api.types import is_numeric_dtype
import dash_bootstrap_components as dbc
from . import ids, select_target
from ..assets.style import RadioItemsStyle, ChecklistStyle, TableStyle, DataStyle, CellStyle


def render(app: Dash, x_to_encode: list[str], x_vars: list[str], y_vars: str, split_ratio: float) -> html.Div:
    df = pd.read_parquet('data/data.parquet')
    x = pd.get_dummies(df[x_vars], columns=x_to_encode, drop_first=True)
    y = pd.get_dummies(df[y_vars], drop_first=True) if not is_numeric_dtype(df[y_vars]) else df[y_vars]

    return html.Div(className='app-div',
                    children=[
                    ])
