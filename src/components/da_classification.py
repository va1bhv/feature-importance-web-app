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
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def render(app: Dash, x_to_encode: list[str], x_vars: list[str], y_vars: str, metrics: list[str],
           split_ratio: float) -> html.Div:
    df = pd.read_parquet('data/data.parquet')
    df.dropna(inplace=True)
    x = pd.get_dummies(df[x_vars], columns=x_to_encode, drop_first=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[y_vars])

    logit = LogisticRegression(random_state=22280)
    tree = DecisionTreeClassifier(random_state=22280)
    metrics_logit = LogisticRegression(random_state=22280)
    metrics_tree = DecisionTreeClassifier(random_state=22280)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=22280)

    logit.fit(x, y)
    tree.fit(x, y)
    metrics_logit.fit(x_train, y_train)
    metrics_tree.fit(x_train, y_train)

    gini_train_y_pred = metrics_tree.predict(x_train)
    gini_test_y_pred = metrics_tree.predict(x_test)
    odds_train_y_pred = metrics_logit.predict(x_train)
    odds_test_y_pred = metrics_logit.predict(x_test)

    gini_cr_df = pd.DataFrame(classification_report(y_test, gini_test_y_pred, output_dict=True,
                                                    target_names=encoder.inverse_transform([0, 1])))
    gini_cr_df = gini_cr_df.drop(columns=['accuracy', 'macro avg']).reset_index().rename(columns={'index': 'Metrics'})
    odds_cr_df = pd.DataFrame(classification_report(y_test, odds_test_y_pred, output_dict=True,
                                                    target_names=encoder.inverse_transform([0, 1])))
    odds_cr_df = odds_cr_df.drop(columns=['accuracy', 'macro avg']).reset_index().rename(columns={'index': 'Metrics'})

    odds_cr_df['Metrics'] = odds_cr_df['Metrics'].apply(lambda n: int(n) if type(n) != str else n)
    gini_cr_df['Metrics'] = gini_cr_df['Metrics'].apply(lambda n: int(n) if type(n) != str else n)
    print(odds_cr_df.to_dict('records'))

    gini_accuracy_train = accuracy_score(y_train, gini_train_y_pred)
    gini_accuracy_test = accuracy_score(y_test, gini_test_y_pred)
    odds_accuracy_train = accuracy_score(y_train, odds_train_y_pred)
    odds_accuracy_test = accuracy_score(y_test, odds_test_y_pred)

    gini_cm_df = pd.DataFrame(confusion_matrix(y_test, gini_test_y_pred, normalize='true'),
                              index=encoder.inverse_transform(range(encoder.classes_.shape[0])),
                              columns=encoder.inverse_transform(range(encoder.classes_.shape[0]))).reset_index().rename(
        columns={'index': 'Actual/Predicted'})
    odds_cm_df = pd.DataFrame(confusion_matrix(y_test, odds_test_y_pred, normalize='true'),
                              index=encoder.inverse_transform(range(encoder.classes_.shape[0])),
                              columns=encoder.inverse_transform(range(encoder.classes_.shape[0]))).reset_index().rename(
        columns={'index': 'Actual/Predicted'})
    percentage = dash_table.FormatTemplate.percentage(2)

    odds = 1 - np.exp(logit.coef_[0])
    gini = tree.feature_importances_
    gini_df = pd.DataFrame({'Variable': x.columns, 'Gini': gini}).sort_values(by='Gini', key=np.abs,
                                                                              ascending=False).reset_index().rename(
        columns={'index': 'Actual/Predicted'})
    odds_df = pd.DataFrame({'Variable': x.columns, 'Odds': odds}).sort_values(by='Odds', key=np.abs,
                                                                              ascending=False).reset_index().rename(
        columns={'index': 'Actual/Predicted'})

    gini_table = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=gini_df.to_dict('records'),
            columns=[{'name': 'Variable', 'id': 'Variable'},
                     {'name': 'Gini', 'id': 'Gini', 'type': 'numeric', 'format': {'specifier': '.4f'}}],
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

    gini_cm = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=gini_cm_df.to_dict('records'),
            columns=[
                {'name': 'Actual/Predicted', 'id': 'Actual/Predicted'},
                {'name': f'{encoder.inverse_transform([0])[0]}', 'id': f'{encoder.inverse_transform([0])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': f'{encoder.inverse_transform([1])[0]}', 'id': f'{encoder.inverse_transform([1])[0]}',
                 'type': 'numeric', 'format': percentage}
            ],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle(textAlign='center')),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Actual/Predicted'},
                 'textAlign': 'center'},
                {'if': {'column_id': 'Actual/Predicted'},
                 'backgroundColor': 'light grey'},
            ]
        )
    ])
    odds_cm = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=odds_cm_df.to_dict('records'),
            columns=[
                {'name': 'Actual/Predicted', 'id': 'Actual/Predicted'},
                {'name': f'{encoder.inverse_transform([0])[0]}', 'id': f'{encoder.inverse_transform([0])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': f'{encoder.inverse_transform([1])[0]}', 'id': f'{encoder.inverse_transform([1])[0]}',
                 'type': 'numeric', 'format': percentage}
            ],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle(textAlign='center')),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Actual/Predicted'},
                 'textAlign': 'center'},
                {'if': {'column_id': 'Actual/Predicted'},
                 'backgroundColor': 'light grey'},
            ]
        )
    ])

    gini_cr = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=gini_cr_df.to_dict('records'),
            columns=[
                {'name': 'Metrics', 'id': 'Metrics'},
                {'name': f'{encoder.inverse_transform([0])[0]}', 'id': f'{encoder.inverse_transform([0])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': f'{encoder.inverse_transform([1])[0]}', 'id': f'{encoder.inverse_transform([1])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': 'weighted avg', 'id': 'weighted avg',
                 'type': 'numeric', 'format': percentage}
            ],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle(textAlign='center')),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Actual/Predicted'},
                 'textAlign': 'center'},
                {'if': {'column_id': 'Actual/Predicted'},
                 'backgroundColor': 'light grey'},
            ]
        )
    ])
    odds_cr = html.Div(className='app-table-div', children=[
        dash_table.DataTable(
            data=odds_cr_df.to_dict('records'),
            columns=[
                {'name': 'Metrics', 'id': 'Metrics'},
                {'name': f'{encoder.inverse_transform([0])[0]}', 'id': f'{encoder.inverse_transform([0])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': f'{encoder.inverse_transform([1])[0]}', 'id': f'{encoder.inverse_transform([1])[0]}',
                 'type': 'numeric', 'format': percentage},
                {'name': 'weighted avg', 'id': 'weighted avg',
                 'type': 'numeric', 'format': percentage}
            ],
            style_table=asdict(TableStyle()),
            style_data=asdict(DataStyle(textAlign='center')),
            style_cell=asdict(CellStyle(textAlign='center')),
            style_data_conditional=[
                {'if': {'column_id': 'Actual/Predicted'},
                 'textAlign': 'center'},
                {'if': {'column_id': 'Actual/Predicted'},
                 'backgroundColor': 'light grey'},
            ]
        )
    ])

    gini_accuracy_col = None
    gini_confusion_matrix_col = None
    gini_classification_report_col = None
    odds_accuracy_col = None
    odds_confusion_matrix_col = None
    odds_classification_report_col = None

    if 'acc' in metrics:
        gini_accuracy_col = dbc.Col(children=[
            html.H5("Model Accuracy", className="card-title"),
            html.H6("Decision Tree Classifier", className="card-subtitle"),
            html.P(
                f"Accuracy of the model during training is {gini_accuracy_train * 100:.2f}%, "
                f"while the accuracy on unseen (test) data is {gini_accuracy_test * 100:.2f}%."
                f"on {split_ratio * 100:.0f}% of the data.",
                className="card-text",
            )
        ],
            align='start',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div',
        )

        odds_accuracy_col = dbc.Col(children=[
            html.H5("Model Accuracy", className="card-title"),
            html.H6("Logistic Regression Classifier", className="card-subtitle"),
            html.P(
                f"Accuracy of the model during training is {odds_accuracy_train * 100:.2f}%, "
                f"while the accuracy on unseen (test) data is {odds_accuracy_test * 100:.2f}%."
                f"on {split_ratio * 100:.0f}% of the data.",
                className="card-text", style={'width': 'auto'}
            )
        ],
            align='start',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div',
        )

    if 'cm' in metrics:
        gini_confusion_matrix_col = dbc.Col(children=[
            html.H5("Confusion Matrix", className="card-title"),
            dbc.Col([gini_cm])
        ],
            align='center',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div'
        )
        odds_confusion_matrix_col = dbc.Col(children=[
            html.H5("Confusion Matrix", className="card-title"),
            dbc.Col([odds_cm])
        ],
            align='center',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div'
        )

    if 'cr' in metrics:
        gini_classification_report_col = dbc.Col(children=[
            html.H5("Classification Report", className="card-title"),
            dbc.Col([gini_cr]),
        ],
            align='end',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div'
        )
        odds_classification_report_col = dbc.Col(children=[
            html.H5("Classification Report", className="card-title"),
            dbc.Col([odds_cr]),
        ],
            align='end',
            style={"width": "auto", 'maxWidth': '50%', 'minWidth': '33%'},
            className='app-div'
        )

    gini_metrics_row = dbc.Row([
        gini_accuracy_col,
        gini_confusion_matrix_col,
        gini_classification_report_col
    ],
        align='center',
        # fluid=True,
        className="g-0",
    )
    odds_metrics_row = dbc.Row([
        odds_accuracy_col,
        odds_confusion_matrix_col,
        odds_classification_report_col
    ],
        align='center',
        # fluid=True,
        className="g-0",
    )

    gini_layout = dbc.Row(
        children=[
            dbc.Row(children=[
                dbc.Col(children=[gini_table]),
                dbc.Col(children=[gini_graph])
            ],
                align='center'
            ),
            gini_metrics_row
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
            odds_metrics_row
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
