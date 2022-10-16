from dash import Dash, html, dcc, no_update
import dash_bootstrap_components as dbc
from . import upload_data, select_target, onehotencode_choose, da_classification, da_continuous
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import datetime
from dataclasses import asdict
from io import StringIO
import sys
import pandas as pd

from ..assets import components

from . import ids


def create_layout(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.UPLOAD_CONTAINER, 'children'),
         Output(ids.TITLE, 'children')],
        Input(ids.UPLOAD_HEADING, 'children'),
        State(ids.UPLOAD_DATA, 'filename'),
        State(ids.UPLOAD_DATA, 'last_modified'),
        State(ids.TITLE, 'children'),
    )
    def make_title_upload(_, filename: str, date: float, title: list) -> (html.Div, html.Div):
        if filename is not None:
            return html.Div(), \
                   html.Div(className='title', children=[
                       html.Div(className='left-title', children=[html.H4(filename),
                                                                  html.Div(
                                                                      children=[
                                                                          f"Last modified: {datetime.datetime.fromtimestamp(date).strftime(r'%d/%m/%Y, %I:%M:%S %p')}"])]),
                       html.Div(className='right-title', children=[
                           components.upload_form,
                       ])
                   ])
        else:
            raise PreventUpdate

    @app.callback(
        Output(ids.LAYOUT, 'children'),
        Input(ids.PROCEED_BUTTON, 'n_clicks'),
    )
    def proceed_upload_select(n: int) -> html.Div:
        if n % 2 != 0:
            return select_target.render(app)
        else:
            raise PreventUpdate

    @app.callback(
        Output(ids.XY_STATUS, 'children'),
        Input(ids.PROCEED_SELECTION_BUTTON, 'n_clicks'),
        State(ids.X_DROPDOWN, 'value'),
        State(ids.Y_DROPDOWN, 'value'),
    )
    def check_selection(n: int, x: list, y: str) -> html.Div:
        if y in x and n > 0:
            return html.Div(
                className='error',
                children=['Target variable cannot be a part of the predictors. Choose again.']
            )
        else:
            return html.Div(children=[])

    @app.callback(
        [Output(ids.METRIC_CONTAINER, 'children'),
         Output(ids.NEW_TITLE, 'children')],
        [Input(ids.PROCEED_SELECTION_BUTTON, 'n_clicks'),
         Input(ids.XY_STATUS, 'children')],
        State(ids.X_DROPDOWN, 'value'),
        State(ids.Y_DROPDOWN, 'value'),
        State(ids.TYPE_OF_MODEL, 'value'),
    )
    def choose_metrics_renderer(n_clicks: int, status: html.Div, x: list[str], y: str, model: str) -> (
            html.Div, html.Div):
        if n_clicks > 0 and status['props']['children'] == []:

            return html.Div([onehotencode_choose.render(app, x, y, model),
                             dbc.Button(id=ids.PROCESS, children=['Process'], outline=True, color='primary')]), \
                   html.Div(children=[html.H3(f'{model} model'),
                                      html.Div(children=[f'{y} ~ f({", ".join(x)})'])])
        else:
            raise PreventUpdate

    @app.callback(
        Output(ids.NEXT, 'children'),
        Input(ids.PROCESS, 'n_clicks'),
        State(ids.X_TABLE, 'selected_rows'),
        State(ids.X_TABLE, 'data'),
        State(ids.X_DROPDOWN, 'value'),
        State(ids.Y_DROPDOWN, 'value'),
        State(ids.TYPE_OF_MODEL, 'value'),
        State(ids.ACCURACY_CHECKLIST, 'value'),
        State(ids.TEST_SPLIT_RATIO, 'value'),
    )
    def one_hot_encoder(n: int, cols: list[int], rows: list[str], x: list[str], y: str, model: str, metrics: list[str],
                        split_ratio: float) -> html.Div:
        try:
            if n > 0:
                data = pd.DataFrame(rows).loc[cols, 'Column'].tolist()
                if model == 'Classification':
                    return da_classification.render(app, data, x, y, metrics, split_ratio)
                else:
                    return da_continuous.render(app, data, x, y, split_ratio)
            else:
                raise PreventUpdate
        except TypeError:
            raise PreventUpdate

    @app.callback(
        Output(ids.COLLAPSE, "is_open"),
        [Input(ids.COLLAPSE_BUTTON, "n_clicks")],
        [State(ids.COLLAPSE, "is_open")],
    )
    def toggle_collapse(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output(ids.BODY, 'children'),
        Input(ids.SELECT_METRICS, 'n_clicks'),
        State(ids.ACCURACY_CHECKLIST, 'value'),
        State(ids.TEST_SPLIT_RATIO, 'value'),
    )
    def render_body(n: int, metrics: list[str], split_ratio: float, ) -> html.Div:
        return html.Div(['Worked'])

    return html.Div(id=ids.FIRST,
                    children=[
                        html.Div(id=ids.NEXT,
                                 children=[
                                     html.Div(
                                         className='app-div',
                                         id=ids.LAYOUT,
                                         children=[
                                             html.Div(id=ids.TITLE, children=[html.H1(app.title)]),
                                             html.Div(
                                                 id=ids.DIV_LAYOUT,
                                                 className='upload-container',
                                                 children=[upload_data.render(app)],
                                             ),
                                             html.Div(id=ids.THROWAWAY, children=[])
                                         ]
                                     )]
                                 ),
                        # html.Div(
                        #     [
                        #         html.Hr(),
                        #         dbc.Button('Reload', id=ids.RELOAD, color='secondary', outline=True, size='md',)
                        #     ],
                        #     className='d-grid gap-2 col-10 mx-auto'
                        # )
                    ]
                    )
