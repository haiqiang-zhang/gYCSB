import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import os
import importlib
import inspect

def create_results_file_explorer(result_options, prefix="results"):
    """Create the file explorer sidebar for results page (Modern Bootstrap-style sidebar)"""
    prefix = f"{prefix}-" if prefix else ""
    
    # Convert result_options to list of filenames if it's a list of dicts
    if result_options and isinstance(result_options[0], dict):
        file_list = [opt.get("label", opt.get("value", "")) for opt in result_options]
    else:
        file_list = result_options if result_options else []
    
    # Create file list items
    file_items = []
    for i, filename in enumerate(file_list):
        file_items.append(
            dbc.Button(
                [
                    html.I(className="bi bi-file-earmark-text me-2", 
                          style={"font-size": "16px", "color": "#0d6efd"}),
                    html.Span(
                        filename,
                        className="file-name-text"
                    )
                ],
                id={"type": "file-row", "index": i, "filename": filename},
                color="link",
                className="file-item-button",
            )
        )
    
    # File list section with Bootstrap badge
    file_list_section = html.Div([
        html.Div([
            html.I(className="bi bi-folder2-open me-2", style={"font-size": "14px"}),
            html.Span("FILE EXPLORER"),
            dbc.Badge(
                str(len(file_list)),
                color="primary",
                pill=True,
                className="ms-2"
            ) if file_list else None
        ], className="result-explorer-header"),
        html.Div(
            file_items if file_items else [
                html.Div([
                    html.I(className="bi bi-inbox text-muted", 
                          style={"font-size": "32px", "margin-bottom": "8px"}),
                    html.Div("No files available", className="text-muted")
                ], className="empty-state")
            ],
            id=f"{prefix}file-list",
            className="file-list-container"
        ),
    ], className="mb-3")
    
    # Actions section with improved buttons
    actions_section = html.Div([
        html.Div([
            html.I(className="bi bi-lightning-charge me-2", style={"font-size": "14px"}),
            html.Span("QUICK ACTIONS")
        ], className="result-explorer-header"),
        html.Div([
            dcc.Upload(
                id=f'{prefix}upload-results',
                multiple=False,
                children=dbc.Button(
                    [
                        html.I(className="bi bi-cloud-upload me-2"),
                        html.Span("Upload File")
                    ],
                    color="primary",
                    outline=True,
                    className="result-action-button",
                    size="sm"
                )
            ),
            dbc.Button(
                [
                    html.I(className="bi bi-plus-circle me-2"),
                    html.Span("New File")
                ],
                id=f"{prefix}create-results",
                color="success",
                outline=True,
                className="result-action-button",
                size="sm"
            )
        ], className="result-actions-container")
    ])
    
    return html.Div([
        html.Div([
            file_list_section,
            html.Hr(className="result-explorer-divider"),
            actions_section
        ], className="explorer-content"),
        # Store to maintain compatibility with existing callbacks
        dcc.Store(id=f"{prefix}result-files", data=None),
        # Store for file list
        dcc.Store(id=f"{prefix}file-list-store", data=file_list)
    ])

def get_available_chart_types():
    """Scan the charts directory and return available chart types"""
    chart_types = []
    charts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'charts')
    
    for file in os.listdir(charts_dir):
        if file.endswith('.py') and not file.startswith('__'):
            module_name = file[:-3]  # Remove .py extension
            try:
                module = importlib.import_module(f'visualizer.charts.{module_name}')
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.endswith('Chart'):
                        # Extract the prefix (e.g., 'MPL' from 'MPLLineChart')
                        prefix = name[:-5]  # Remove 'Chart' suffix
                        chart_types.append({
                            "label": f"{prefix} ({module_name})",
                            "value": module_name
                        })
            except Exception as e:
                print(f"Error loading chart module {module_name}: {str(e)}")
    
    return chart_types


def create_results_card():
    """Create the results display card with modern Bootstrap styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-table me-2", style={"font-size": "20px"}),
                html.Span("Results Data", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-primary text-white"),
        dbc.CardBody([
            # Action buttons with icons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="bi bi-save me-2"),
                            "Save Changes"
                        ], 
                        id="save-changes-button", 
                        color="success",
                        size="sm"),
                        dbc.Button([
                            html.I(className="bi bi-check-all me-2"),
                            "Select All"
                        ], 
                        id="select-all-button", 
                        color="primary",
                        outline=True,
                        size="sm"),
                        dbc.Button([
                            html.I(className="bi bi-x-circle me-2"),
                            "Clear All"
                        ], 
                        id="clear-all-button", 
                        color="secondary",
                        outline=True,
                        size="sm")
                    ], size="sm")
                ], width=12)
            ], className="mb-3"),
            
            # Column selector with icon
            dbc.Row([
                dbc.Col([
                    html.Label([
                        html.I(className="bi bi-columns-gap me-2"),
                        "Display Columns"
                    ], className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id="display-columns",
                        multi=True,
                        clearable=False,
                        searchable=True,
                        className="custom-dropdown"
                    )
                ], width=12)
            ], className="mb-3"),
            
            # Message/Hint container (outside scrollable area for better visibility)
            html.Div(
                id="results-message",
                className="mb-3",
                style={
                    "minHeight": "0px",
                    "transition": "all 0.3s ease-in-out"
                }
            ),
            
            # Scrollable table container
            html.Div(
                html.Div(id="results-table"),
                className="results-table-container"
            )
        ])
    ], className="mb-4 shadow-sm")

def create_visualization_config_card():
    """Create the visualization configuration card with modern styling"""
    chart_types = get_available_chart_types()
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-gear-fill me-2", style={"font-size": "20px"}),
                html.Span("Visualization Configuration", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-dark text-white"),
        dbc.CardBody([
            # Chart type selector
            dbc.Row([
                dbc.Col([
                    html.Label([
                        html.I(className="bi bi-bar-chart-line me-2"),
                        "Chart Type"
                    ], className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id="visualization-type",
                        options=chart_types,
                        value=chart_types[0]['value'] if chart_types else None,
                        className="custom-dropdown"
                    ),
                    dcc.Store(id='visualization-type-store')
                ], width=12)
            ], className="mb-3"),
            
            # Axis configuration in a card-like section
            html.Div([
                html.H6([
                    html.I(className="bi bi-rulers me-2"),
                    "Axis Configuration"
                ], className="mb-3 text-primary"),
                dbc.Row([
                    dbc.Col([
                        html.Label("X-Axis", className="fw-bold small"),
                        dcc.Dropdown(id="x-axis-dropdown",
                                     searchable=True,
                                     className="custom-dropdown")
                    ], md=4),
                    dbc.Col([
                        html.Label("Y-Axis", className="fw-bold small"),
                        dcc.Dropdown(id="y-axis-dropdown",
                                     searchable=True,
                                     className="custom-dropdown")
                    ], md=4),
                    dbc.Col([
                        html.Label("Group By", className="fw-bold small"),
                        dcc.Dropdown(id="group-by-dropdown",
                                     multi=True,
                                     searchable=True,
                                     className="custom-dropdown")
                    ], md=4)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label([
                            html.I(className="bi bi-aspect-ratio me-2"),
                            "X-Axis Scale"
                        ], className="fw-bold small"),
                        dcc.Dropdown(
                            id="x-axis-scale",
                            options=[
                                {"label": "Linear", "value": "linear"},
                                {"label": "Log2", "value": "log2"}
                            ],
                            value="linear",
                            className="custom-dropdown"
                        )
                    ], md=4)
                ])
            ], className="p-3 mb-3 bg-light rounded"),
            
            # Export button
            html.Div([
                dcc.Download(id="download-pdf"),
                dbc.Button([
                    html.I(className="bi bi-file-earmark-pdf me-2"),
                    "Export as PDF"
                ], id="save-pdf", color="success", className="w-100", size="sm")
            ])
        ])
    ], className="mb-4 shadow-sm")

def create_chart_card():
    """Create the chart display card with modern styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-graph-up-arrow me-2", style={"font-size": "20px"}),
                html.Span("Chart Visualization", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-success text-white"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className="bi bi-graph-up fs-1 mb-3 text-primary"),
                    html.H4("No Chart Available", className="mb-2 text-secondary"),
                    html.P([
                        html.I(className="bi bi-info-circle me-2"),
                        "Configure the visualization options above to generate your chart"
                    ], className="text-muted")
                ], className="text-center py-5")
            ], id="chart-container", className="chart-display-area")
        ])
    ], className="shadow-sm")

def create_results_table(results, select_all=True, display_columns=None):
    """Create a table with delete buttons and checkboxes for each row"""
    if isinstance(results, pd.DataFrame):
        df = results
    else:
        df = pd.DataFrame(results)
        
    if 'operation_details' in df.columns:
        df = df.drop(columns=['operation_details'])
    
    if display_columns:
        # Filter to only include columns that exist in the dataframe
        valid_columns = [col for col in display_columns if col in df.columns]
        if valid_columns:
            df = df[valid_columns]
        # If no valid columns, use all columns
        elif len(df.columns) > 0:
            df = df
        
    header = html.Thead(html.Tr([
        html.Th("Select"),
        html.Th("Actions")
    ] + [html.Th(col) for col in df.columns]))
    
    body = html.Tbody([
        html.Tr([
            html.Td(
                dcc.Checklist(
                    id={"type": "row-select", "index": i},
                    options=[{"label": "", "value": "selected"}],
                    value=["selected"] if select_all else [],
                    className="mb-0"
                )
            ),
            html.Td(
                dbc.Button(
                    "Delete",
                    id={"type": "delete-row", "index": i},
                    color="danger",
                    size="sm"
                )
            )
        ] + [
            html.Td(str(df.iloc[i][col])) for col in df.columns
        ]) for i in range(len(df))
    ])
    
    return dbc.Table([header, body], striped=True, bordered=True, hover=True) 


