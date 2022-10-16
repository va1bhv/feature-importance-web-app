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

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def render(app: Dash, x_to_encode: list[str], x_vars: list[str], y_vars: str, metrics: list[str],
           split_ratio: float) -> html.Div:
    df = pd.read_parquet('data/data.parquet')
    x = pd.get_dummies(df[x_vars], columns=x_to_encode, drop_first=True)
    y = pd.get_dummies(df[y_vars], drop_first=True) if not is_numeric_dtype(df[y_vars]) else df[y_vars]

    logit = LogisticRegression(random_state=22280)
    tree = DecisionTreeClassifier(random_state=22280)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=22280)

    logit.fit(x, y)
    tree.fit(x, y)

    odds = 1 - np.exp(logit.coef_[0])
    gini = tree.feature_importances_
    gini_df = pd.DataFrame({'Variable': x.columns, 'Gini': gini}).sort_values(by='Gini', key=np.abs, ascending=False)
    odds_df = pd.DataFrame({'Variable': x.columns, 'Odds': odds}).sort_values(by='Odds', key=np.abs, ascending=False)

    gini_table = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=gini_df.to_dict('records'),
            columns=[{'name': 'Variable', 'id': 'Variable'},
                     {'name': 'Gini', 'id': 'Gini', 'type': 'numeric',
                      'format': {'specifier': '.4f'}}],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle()),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Gini'},
                 'width': '30%'},
                {'if': {'column_id': 'Gini'},
                 'textAlign': 'right'}
            ]
        )
    ])
    odds_table = html.Div(className='app-table-div', children=[dash_table.DataTable(
        data=odds_df.to_dict('records'),
        columns=[{'name': 'Variable', 'id': 'Variable'},
                 {'name': 'Impact to final Odds', 'id': 'Odds', 'type': 'numeric',
                  'format': {'specifier': '.4f'}}],
        style_table=asdict(TableStyle()),
        style_data=asdict(DataStyle()),
        style_cell=asdict(CellStyle(textAlign='center')),
        style_data_conditional=[
            {'if': {'column_id': 'Odds'},
             'width': '30%'},
            {'if': {'column_id': 'Odds'},
             'textAlign': 'right'},
        ]
    )])

    gini_bar = px.bar(gini_df, y='Variable', x='Gini', template='plotly_white', orientation='h')
    gini_bar.update_layout(yaxis=dict(autorange="reversed"))
    gini_graph = dcc.Graph(figure=gini_bar)

    odds_bar = px.bar(odds_df, y='Variable', x='Odds', template='plotly_white', orientation='h')
    odds_bar.update_layout(yaxis=dict(autorange="reversed"))
    odds_graph = dcc.Graph(figure=odds_bar)

    gini_layout = dbc.Row(
        children=[
            dbc.Row(children=[
                dbc.Col(children=[gini_table]),
                dbc.Col(children=[gini_graph])
            ],
                align='center'
            ),
            html.Div(className='app-div', children=['Hello Gini'])
        ]
    )
    odds_layout = dbc.Row(
        children=[
            dbc.Row(children=[
                dbc.Col(children=[odds_table]),
                dbc.Col(children=[odds_graph])
            ],
                align='center'
            ),
            html.Div(className='app-div', children=['Hello Odds'])
        ]
    )

    odds_tab = dbc.Tab(
        label='Impact to final Odds',
        children=[odds_layout]
    )
    gini_tab = dbc.Tab(
        label='Gini Index',
        children=[gini_layout]
    )

    tabs = dbc.Tabs(
        children=[
            odds_tab,
            gini_tab
        ]
    )

    return html.Div(className='app-div',
                    children=[
                        tabs
                    ])
