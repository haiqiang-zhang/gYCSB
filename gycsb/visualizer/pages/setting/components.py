import dash_bootstrap_components as dbc
from dash import html
import os
from gycsb.ConfigLoader import load_system_setting

def create_process_management_card():
    """Create a card for process management with modern styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-cpu me-2", style={"font-size": "20px"}),
                html.Span("Process Management", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-secondary text-white"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className="bi bi-info-circle me-3 text-muted", style={"font-size": "48px"}),
                    html.Div([
                        html.H5("No functionality available at this time", className="text-muted mb-2"),
                        html.P("Process management features will be added in a future update", 
                              className="text-muted small mb-0")
                    ])
                ], className="d-flex align-items-center justify-content-center p-4 bg-light rounded text-center")
            ])
        ])
    ], className="mb-4 shadow-sm")

def create_directory_config_card():
    """Create a card for directory configuration with modern styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-folder-fill me-2", style={"font-size": "20px"}),
                html.Span("Directory Configuration", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-info text-white"),
        dbc.CardBody([
            html.Div([
                html.H6([
                    html.I(className="bi bi-folder2-open me-2"),
                    "Operation File Directory"
                ], className="mb-3 text-info"),
                
                html.Div([
                    html.Label([
                        html.I(className="bi bi-file-code me-2"),
                        "Directory Path"
                    ], className="fw-bold small mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText(html.I(className="bi bi-folder")),
                        dbc.Input(
                            id="operation-file-dir",
                            type="text",
                            value=load_system_setting().get('operation_file_dir', ''),
                            placeholder="Enter path to operation files directory"
                        )
                    ], className="mb-3"),
                    
                    html.Div([
                        html.I(className="bi bi-info-circle me-2 text-primary"),
                        html.Span("This directory contains the operation files used for benchmarking", 
                                 className="text-muted small")
                    ], className="d-flex align-items-start p-3 bg-light rounded")
                ])
            ])
        ])
    ], className="mb-4 shadow-sm")

def create_system_info_card():
    """Create a card for system information with modern styling"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="bi bi-info-circle-fill me-2", style={"font-size": "20px"}),
                html.Span("System Information", style={"font-size": "18px", "font-weight": "600"})
            ], className="d-flex align-items-center")
        ], className="bg-success text-white"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="bi bi-server me-3 text-primary", style={"font-size": "24px"}),
                        html.Div([
                            html.Div("Server Status", className="text-muted small"),
                            html.Div("Running", className="fw-bold text-success")
                        ])
                    ], className="d-flex align-items-center mb-3 p-3 bg-light rounded"),
                    
                    html.Div([
                        html.I(className="bi bi-hdd-rack me-3 text-primary", style={"font-size": "24px"}),
                        html.Div([
                            html.Div("Current Configuration", className="text-muted small"),
                            html.Div(id="config-path-display", className="fw-bold")
                        ])
                    ], className="d-flex align-items-center p-3 bg-light rounded")
                ])
            ])
        ])
    ], className="mb-4 shadow-sm")