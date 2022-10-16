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
from sklearn.linear_model import LinearRegression
import numpy as np


def render(app: Dash, x_to_encode: list[str], x_vars: list[str], y_vars: str, split_ratio: float) -> html.Div:
    df = pd.read_parquet('data/data.parquet')
    x = pd.get_dummies(df[x_vars], columns=x_to_encode, drop_first=True)
    y = pd.get_dummies(df[y_vars], drop_first=True) if not is_numeric_dtype(df[y_vars]) else df[y_vars]

    regressor = LinearRegression()
    regressor.fit(x, y)
    coef = regressor.coef_
    coef_df = pd.DataFrame({'Variable': x.columns, 'Coefficient': coef}).sort_values(by='Coefficient', key=np.abs, ascending=False)

    coef_table = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=coef_df.to_dict('records'),
            columns=[{'name': 'Variable', 'id': 'Variable'},
                     {'name': 'Coefficient', 'id': 'Coefficient', 'type': 'numeric',
                      'format': {'specifier': '.4f'}}],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle()),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Coefficient'},
                 'width': '30%'},
                {'if': {'column_id': 'Coefficient'},
                 'textAlign': 'right'}
            ]
        )
    ])
    coef_bar = px.bar(coef_df, y='Variable', x='Coefficient', template='plotly_white', orientation='h')
    coef_bar.update_layout(yaxis=dict(autorange="reversed"))
    coef_graph = dcc.Graph(figure=coef_bar)

    coef_layout = dbc.Row(
        children=[
            dbc.Row(children=[
                dbc.Col(children=[coef_table]),
                dbc.Col(children=[coef_graph])
            ],
                align='center'
            ),
            html.Div(className='app-div', children=['Hello Coef'])
        ]
    )

    return html.Div(className='app-div',
                    children=[
                        coef_layout
                    ])
