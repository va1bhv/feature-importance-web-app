from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import datetime
import io
import pandas as pd
from dataclasses import asdict

from . import ids
from ..assets.style import TableStyle, CellStyle, DataStyle
from ..assets import components


def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.UPLOAD_DATA_OUTPUT, 'children'),
        [Input(ids.UPLOAD_DATA, 'contents'),
         Input(ids.UPLOAD_DATA, 'filename'),
         Input(ids.UPLOAD_DATA, 'last_modified')]
    )
    def update_output(contents: str, filename: str, date: float) -> html.Div:
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            try:
                if 'csv' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    df = pd.read_excel(io.BytesIO(decoded))
                df.to_parquet('data/data.parquet')
            except Exception as e:
                print(e)
                return html.Div(
                    'There was an error processing the file. Please ensure only csv or excel files are uploaded.')

            return html.Div(children=[
                html.Hr(),
                html.H6(id=ids.UPLOAD_HEADING, children=['Uploaded table:']),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table=asdict(TableStyle()),
                    style_data=asdict(DataStyle()),
                    style_cell=asdict(CellStyle(textAlign='center')),
                ),
                html.Hr(),
                html.H6('Table info:'),
                dash_table.DataTable(
                    data=[{'Column': col, 'dtype': dtype} for col, dtype in
                          zip(df.columns, [str(i) for i in df.dtypes])],
                    style_table=asdict(TableStyle(width='50%')),
                    style_data=asdict(DataStyle()),
                    style_cell=asdict(CellStyle(textAlign='center')),
                ),
                dbc.Button(
                    id=ids.PROCEED_BUTTON,
                    children=['Proceed'],
                    className='button',
                    n_clicks=0,
                    outline=True,
                    color='primary'
                ),
            ])

    return html.Div(
        children=[html.Div(id=ids.UPLOAD_CONTAINER,
                           className='upload-container',
                           children=[
                               html.H6('Upload File (excel or csv only)'),
                               html.Div(children=[
                                   components.upload_form
                               ]),
                           ]),
                  html.Div(id=ids.UPLOAD_DATA_OUTPUT)
                  ],

    )
