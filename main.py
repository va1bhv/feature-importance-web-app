from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from src.components.layout import create_layout


def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP], suppress_callback_exceptions=True, prevent_initial_callbacks=True)
    app.title = 'Feature Importances'
    app.layout = create_layout(app)
    app.run(debug=True,)


if __name__ == '__main__':
    main()
