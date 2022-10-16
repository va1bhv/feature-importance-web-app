from dash import Dash, dcc, html, dash_table
import plotly.express as px
from dash.dependencies import Input, Output, State
import base64
import datetime
import io
import pandas as pd
from dataclasses import asdict
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from . import ids, select_target, da_classification
from ..assets.style import RadioItemsStyle, ChecklistStyle, TableStyle, DataStyle, CellStyle


def render(app, x: list[str], y: str, model: str) -> html.Div:
    df = pd.read_parquet('data/data.parquet')
    cols = df[x].columns.to_list()
    cols.append(y)

    one_hot_encode_table = html.Div(
        children=[
            html.H6('Select the variables to be one-hot-encoded and/or label encoded.'),
            dash_table.DataTable(
                id=ids.X_TABLE,
                data=[{'Column': col, 'dtype': dtype} for col, dtype in
                      zip(x, [str(i) for i in df[x].dtypes])],
                style_table=asdict(TableStyle(width='95%')),
                style_data=asdict(DataStyle()),
                style_cell=asdict(CellStyle(textAlign='center')),
                row_selectable='multi',
                selected_rows=[i for i in range(len(x)) if df[x].iloc[:, i].dtypes == 'object'],
                style_cell_conditional=[
                    {'if': {'column_id': 'dtype'},
                     'width': '30%'},
                ]
            )
        ]
    )
    accuracy_options = html.Div(children=[
        html.H5('Select the metrics to be shown.'),
        dbc.Checklist(
            id=ids.ACCURACY_CHECKLIST,
            options=[{'label': i, 'value': j} for i, j in
                     zip(['Accuracy', 'Confusion Matrix', 'Classification Report'], ['acc', 'cm', 'cr'])],
            value=['acc', 'cm', 'cr'],
            inline=True
        )
    ])

    slider = dbc.Col([
        dbc.Row([html.H5(['Choose test-train split ratio.'])]),
        dbc.Row([dcc.Slider(
            id=ids.TEST_SPLIT_RATIO,
            min=0.05, max=0.95, step=0.05,
            value=0.75,
            marks={i / 100: f'{i}%' for i in range(5, 100, 5)},
        )])
    ])

    header = html.Div([
        dbc.Button(id=ids.COLLAPSE_BUTTON, children=['More Options'], n_clicks=0, className='mb-3', color='primary',
                   outline=True),
        dbc.Collapse([dbc.Row([
            dbc.Col([
                accuracy_options
            ]),
            dbc.Col([
                slider
            ])
        ]),
        ],
            is_open=False,
            id=ids.COLLAPSE
        )])

    if model == 'Classification':

        bin_vs_multi = html.Div(children=[
            html.H6(f'Is {y} a binary variable or a multi-class variable?'),
            dcc.RadioItems(
                id=ids.BINARY_VS_MULTICLASS_RADIO,
                className='checklist',
                options=['Binary', 'Multi-class'],
                value='Binary',
                labelStyle=asdict(RadioItemsStyle(display='inline-block'))
            )
        ])

        pie = px.pie(df, names=y, title=f'Distribution of categories in {y}',
                     color_discrete_sequence=px.colors.sequential.Tealgrn_r)
        pie.update_traces(textposition='inside', textinfo='percent+label',
                          marker=dict(line=dict(color='#1B1B1B', width=0.5)))
        graph = dcc.Graph(
            ids.GRAPH,
            figure=pie
        )

        return html.Div(
            children=[
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='left',
                            children=[graph]
                        ),
                        html.Div(
                            className='right',
                            children=one_hot_encode_table
                        )
                    ]
                ),
                header,
                html.Hr()
            ])

    else:
        hist = px.histogram(df, x=y)
        graph = dcc.Graph(
            ids.GRAPH,
            figure=hist
        )
        return html.Div(
            children=[
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='left',
                            children=[graph]
                        ),
                        html.Div(
                            className='right',
                            children=one_hot_encode_table
                        )
                    ]
                ),
                html.Div([
                    dbc.Button(id=ids.COLLAPSE_BUTTON, children=['More Options'], n_clicks=0, className='mb-3',
                               color='primary',
                               outline=True),
                    dbc.Collapse([dbc.Row([
                        dbc.Col([
                            slider,
                            html.Div(id=ids.ACCURACY_CHECKLIST)
                        ])
                    ]),
                    ],
                        is_open=False,
                        id=ids.COLLAPSE
                    )])
            ])
