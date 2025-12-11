from dash import register_page
import dash_bootstrap_components as dbc
from dash import html, dcc
from visualizer.pages.setting.components import (
    create_process_management_card,
    create_directory_config_card,
    create_system_info_card
)
from visualizer.callbacks.setting_callbacks import register_setting_callbacks


def create_setting_layout():
    """Create the setting page layout with modern Bootstrap styling"""
    return dbc.Container([
        # Hero-style page header
        html.Div([
            html.H1([
                html.I(className="bi bi-gear-fill me-3 text-primary"),
                "Settings"
            ], className="page-title", style={"paddingTop": "20px"}),
            html.P(
                "Configure system settings, manage processes, and view system information",
                className="text-muted lead"
            )
        ], className="mb-4"),
        

        
        # Directory Configuration Section
        create_directory_config_card(),
        
        # System Information Section
        create_system_info_card(),
        
        
        # Process Management Section
        create_process_management_card(),
        
    ], fluid=True, style={"padding": "0 30px 20px 30px", "background-color": "#f8f9fa", "minHeight": "calc(100vh - 56px)"})


# Set the layout
layout = create_setting_layout() 
    
# Register this page
register_page(__name__, path='/setting', title="Setting", layout=layout)

# Register callbacks
register_setting_callbacks()
