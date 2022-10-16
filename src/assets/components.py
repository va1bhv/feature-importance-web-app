from dash import Dash, dcc, html, dash_table

from dataclasses import asdict
from ..components import ids
from ..assets.style import UploadStyle

upload_form = dcc.Upload(
    id=ids.UPLOAD_DATA,
    className='upload-assets',
    filename=None,
    children=html.Div([
        'Drag and drop or ',
        html.A('Select File', className='a')]),
    multiple=False,
    style=asdict(UploadStyle()),
)
