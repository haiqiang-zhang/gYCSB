from dash import register_page
from visualizer.VisualizerModel import VisualizerModel
import dash_bootstrap_components as dbc
from dash import html, dcc
from visualizer.pages.results.components import (
    create_results_card,
    create_visualization_config_card,
    create_chart_card,
    create_results_file_explorer
)

from visualizer.callbacks.visualization_callbacks import register_visualization_callbacks
from visualizer.callbacks.results_callbacks import register_results_callbacks

def create_results_layout(result_options):
    """Create the results and visualization page layout with responsive design
    - Mobile: Stacked layout
    - Desktop: Fixed-width sidebar (280px) + flexible content area with resizable divider
    """
    return dbc.Container([
        # Responsive layout: stacked on mobile, fixed sidebar on desktop
        html.Div([
            # Left column: File Explorer (VSCode-style sidebar with adjustable width)
            html.Div(
                html.Div(
                    create_results_file_explorer(result_options, prefix="results"),
                    className="left-scroll-container",
                    id="left-scroll-container"
                ),
                className="file-explorer-col",
                id="file-explorer-col"
            ),
            
            # Resizer handle (VSCode-style draggable divider)
            html.Div(
                className="resizer",
                id="resizer",
                **{"data-direction": "horizontal"}
            ),
            
            # Right column: Main content with title (flexible width)
            html.Div(
                html.Div([
                    # Title at the top of right column - Modern Bootstrap style
                    html.Div([
                        html.H1([
                            html.I(className="bi bi-bar-chart-line-fill me-3 text-primary"),
                            "Results & Visualization"
                        ], className="page-title", style={"paddingTop": "20px"}),
                        html.P(
                            "Analyze benchmark results and create custom visualizations",
                            className="text-muted lead"
                        )
                    ], className="mb-4"),
                    
                    # Results Table
                    create_results_card(),
                    
                    # Visualization Configuration
                    create_visualization_config_card(),
                    
                    # Chart
                    create_chart_card(),
                ], 
                className="right-scroll-container",
                id="right-scroll-container"),
                className="main-content-col",
                id="main-content-col"
            )
        ], className="results-row"),
        
        # Store for selected results
        dcc.Store(id='selected-results', data=[]),
        
        # Store for explorer width (persisted in localStorage)
        dcc.Store(id='explorer-width-store', data=280, storage_type='local'),
        
        # Store for per-file configurations (columns and visualization settings)
        dcc.Store(id='file-configs-store', data={}, storage_type='local'),
        
        # Store for last opened file (persisted in localStorage)
        dcc.Store(id='last-opened-file-store', data=None, storage_type='local')
    ], fluid=True, className="results-container") 



# Create visualizer instance and get available results
model = VisualizerModel()


# Set the layout
layout = create_results_layout(model.get_available_results()) 

# Register this page
register_page(__name__, path='/', title="Results & Visualization", layout=layout)

register_visualization_callbacks(model)
register_results_callbacks(model)