from dash import Dash, dcc, html, dash_table
from dataclasses import asdict
import pandas as pd
import dash_bootstrap_components as dbc
from . import ids
from ..assets.style import RadioItemsStyle
from dash.exceptions import PreventUpdate
from . import da_classification
from dash.dependencies import Output, Input, State


def render(app):
    cols = pd.read_parquet('data/data.parquet').columns.to_list()
    y_selection = html.Div(
        className='y-dropdown',
        children=[
            html.H6('Select target variable'),
            dcc.Dropdown(
                id=ids.Y_DROPDOWN,
                options=cols,
                value=cols[-1],
                multi=False
            ),
            dbc.RadioItems(
                id=ids.TYPE_OF_MODEL,
                options=[{'label': i, 'value': j} for i, j in zip(['Categorical', 'Continuous'], ['Classification', 'Continuous'])],
                value='Classification',
            ),

        ]
    )
    x_selection = html.Div(
        className='x-dropdown',
        children=[
            html.H6('Select predictors'),
            dcc.Dropdown(
                id=ids.X_DROPDOWN,
                options=cols,
                value=cols,
                multi=True
            )
        ]
    )

    proceed_selection_button = dbc.Button(
        className='button',
        id=ids.PROCEED_SELECTION_BUTTON,
        children=['Proceed'],
        n_clicks=0,
        outline=True,
        color='primary'
    )

    status = html.Div(
        id=ids.XY_STATUS,
        children=[]
    )

    metric = html.Div(
        id=ids.METRIC_CONTAINER,
        children=[]
    )

    return html.Div(children=[
        html.Div(id=ids.NEW_TITLE, children=[html.H3('Choose variables')]),
        html.Hr(),
        html.Div(id=ids.XY_CONTAINER,
                 children=[
                     html.Div(
                         className='row',
                         children=[y_selection, x_selection]
                     ),
                     proceed_selection_button,
                     status,
                     html.Hr(),
                     metric,
                     html.Div(id=ids.THROWAWAY, children=[])
                 ]
                 )
    ])
