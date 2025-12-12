import dash_bootstrap_components as dbc
from dash import html, dcc


def create_navigation(pathname="/"):
    """Create the navigation bar component with active page highlighting
    
    Args:
        pathname (str): Current path of the page
    """
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Results & Visualization", href="/", active=pathname == "/")),
            dbc.NavItem(dbc.NavLink("Setting", href="/setting", active=pathname == "/setting")),
        ],
        brand="gYCSB",
        brand_href="/",
        color="black",
        dark=True,
        fixed="top",
        className="mb-4",
        brand_style={"fontWeight": "bold"}
    )


def create_new_results_modal(prefix=""):
    """Create a modal dialog for setting the filename when creating new results"""
    prefix = f"{prefix}-" if prefix else ""
    return dbc.Modal([
        dbc.ModalHeader("Create New Results File"),
        dbc.ModalBody([
            dbc.Input(
                id=f"{prefix}new-filename",
                placeholder="Enter filename (e.g., new_results.json)",
                type="text"
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id=f"{prefix}cancel-create", className="ms-auto", n_clicks=0),
            dbc.Button("Create", id=f"{prefix}confirm-create", className="ms-2", n_clicks=0)
        ])
    ], id=f"{prefix}create-modal", is_open=False)