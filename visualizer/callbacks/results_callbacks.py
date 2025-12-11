import json
import base64
import pandas as pd
import time
from dash import callback, Input, Output, State, callback_context, ALL, no_update
import dash_bootstrap_components as dbc
from dash import html
import os
from ..pages.results.components import create_results_table
from visualizer.VisualizerModel import VisualizerModel

def handle_result_file_load(result_file, upload_contents, model):
    """Handle loading results from file or upload"""
    if result_file:
        file_path = os.path.join(model.results_dir, result_file)
        if model.load_results_from_file(file_path):
            return model.results
        return "Error loading results file"
    elif upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded)
            # Support new format with 'results' field, or old format (direct array)
            if isinstance(data, dict) and 'results' in data:
                model.results = data['results']
            elif isinstance(data, list):
                model.results = data
            else:
                model.results = data
            return model.results
        except Exception as e:
            return f"Error processing uploaded file: {str(e)}"
    return "Please select a results file or upload one"


def get_results_columns(runner):
    df = pd.DataFrame(runner.results)
    if 'operation_details' in df.columns:
        df = df.drop(columns=['operation_details'])
    return df.columns

def save_changes(runner, display_columns):
    """Save current results to JSON file
    
    Returns:
        tuple: (message, table) where message is the status message and table is the results table
    """
    if not runner.current_file:
        message = dbc.Alert([
            html.Div([
                html.I(className="bi bi-exclamation-triangle-fill me-2", style={"fontSize": "20px"}),
                html.Strong("No File Loaded", className="me-2"),
            ], className="d-flex align-items-center mb-2"),
            html.P("Please select or upload a results file before saving.", className="mb-0 ms-4")
        ], color="warning", className="d-flex flex-column", dismissable=True, duration=4000)
        table = create_results_table(runner.results, select_all=True, display_columns=display_columns) if runner.results else "No results available"
        return message, table
        
    try:
        with open(runner.current_file, 'r') as f:
            current_content = json.load(f)
        
        
        is_old_format = False
        # back compatibility with old results format
        if 'results' in current_content:
            current_results = json.dumps(current_content['results'], sort_keys=True)
        else:
            is_old_format = True
            current_results = json.dumps(current_content, sort_keys=True)
        new_results = json.dumps(runner.results, sort_keys=True)
        
        if current_results == new_results:
            message = dbc.Alert([
                html.Div([
                    html.I(className="bi bi-info-circle-fill me-2", style={"fontSize": "20px"}),
                    html.Strong("No Changes Detected", className="me-2"),
                ], className="d-flex align-items-center mb-2"),
                html.P("The current data matches the saved file. Nothing to save.", className="mb-0 ms-4")
            ], color="info", className="d-flex flex-column", dismissable=True, duration=4000)
            table = create_results_table(runner.results, select_all=True, display_columns=display_columns)
            return message, table
            
        with open(runner.current_file, 'w') as f:
            if is_old_format:
                current_content = runner.results
            else:
                current_content['results'] = runner.results
            json.dump(current_content, f, indent=4)
            message = dbc.Alert([
                html.Div([
                    html.I(className="bi bi-check-circle-fill me-2", style={"fontSize": "20px"}),
                    html.Strong("Success!", className="me-2"),
                ], className="d-flex align-items-center mb-2"),
                html.P([
                    "Changes have been saved to ",
                    html.Code(os.path.basename(runner.current_file), className="bg-light px-2 py-1 rounded")
                ], className="mb-0 ms-4")
            ], color="success", className="d-flex flex-column", dismissable=True, duration=4000)
            table = create_results_table(runner.results, select_all=True, display_columns=display_columns)
            return message, table
    except Exception as e:
        message = dbc.Alert([
            html.Div([
                html.I(className="bi bi-x-circle-fill me-2", style={"fontSize": "20px"}),
                html.Strong("Error Saving Changes", className="me-2"),
            ], className="d-flex align-items-center mb-2"),
            html.P([
                "Failed to save changes: ",
                html.Code(str(e), className="bg-light px-2 py-1 rounded text-danger")
            ], className="mb-0 ms-4")
        ], color="danger", className="d-flex flex-column", dismissable=True)
        table = create_results_table(pd.DataFrame(runner.results), select_all=True, display_columns=display_columns)
        return message, table

def _create_file_list_items(file_list, selected_file=None):
    """Helper function to create file list items"""
    from dash import html
    import dash_bootstrap_components as dbc
    file_items = []
    for i, filename in enumerate(file_list):
        is_selected = (filename == selected_file)
        file_items.append(
            dbc.Button(
                [
                    html.Span("📄", style={"margin-right": "8px", "font-size": "14px"}),
                    html.Span(
                        filename,
                        style={
                            "font-size": "13px",
                            "user-select": "none",
                            "overflow": "hidden",
                            "text-overflow": "ellipsis",
                            "white-space": "nowrap",
                            "text-align": "left"
                        }
                    )
                ],
                id={"type": "file-row", "index": i, "filename": filename},
                color="link",
                className="file-item-button" + (" file-item-selected" if is_selected else ""),
                style={
                    "width": "100%",
                    "padding": "6px 8px",
                    "margin": "2px 0",
                    "border-radius": "3px",
                    "cursor": "pointer",
                    "display": "flex",
                    "align-items": "center",
                    "transition": "background-color 0.2s",
                    "text-decoration": "none",
                    "background-color": "#0078d4" if is_selected else "transparent",
                    "border": "none",
                    "color": "white" if is_selected else "inherit",
                    "justify-content": "flex-start"
                }
            )
        )
    
    if not file_items:
        from dash import html
        file_items = [
            html.Div(
                "No files available",
                style={"padding": "10px", "color": "#999", "font-size": "12px", "font-style": "italic"}
            )
        ]
    
    return file_items

def register_results_callbacks(model: VisualizerModel):
    # Callback to restore last opened file on page load
    @callback(
        [Output("results-result-files", "data", allow_duplicate=True),
         Output("results-file-list", "children", allow_duplicate=True),
         Output("last-opened-file-store", "data", allow_duplicate=True)],
        [Input("results-file-list-store", "data")],
        [State("last-opened-file-store", "data"),
         State("results-result-files", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def restore_last_opened_file(file_list, last_opened_file, current_selected_file):
        # If a file is already selected, don't restore
        if current_selected_file:
            return no_update, no_update, no_update
        
        # If no last opened file, don't restore
        if not last_opened_file:
            return no_update, no_update, no_update
        
        # Ensure file_list is available
        if not file_list:
            model.available_results = model.get_available_results()
            file_list = model.available_results
        
        # Check if the last opened file still exists
        if last_opened_file in file_list:
            # File exists, restore it
            file_items = _create_file_list_items(file_list, last_opened_file)
            return last_opened_file, file_items, no_update
        else:
            # File doesn't exist anymore, clear the store
            file_items = _create_file_list_items(file_list, None)
            return None, file_items, None
    
    # Callback to handle file item clicks
    @callback(
        [Output("results-result-files", "data"),
         Output("results-file-list", "children"),
         Output("last-opened-file-store", "data")],
        [Input({"type": "file-row", "index": ALL, "filename": ALL}, "n_clicks")],
        [State({"type": "file-row", "index": ALL, "filename": ALL}, "id"),
         State("results-file-list-store", "data"),
         State("results-result-files", "data")],
        prevent_initial_call=True
    )
    def handle_file_click(n_clicks, file_ids, file_list, current_selected_file):
        ctx = callback_context
        if not ctx.triggered or not n_clicks:
            return no_update, no_update, no_update
        
        # Find which file was clicked
        clicked_file = None
        for i, (clicks, file_id) in enumerate(zip(n_clicks, file_ids)):
            if clicks and clicks > 0:
                clicked_file = file_id.get("filename")
                break
        
        if not clicked_file:
            return no_update, no_update, no_update
        
        # Update file list with selected state
        file_items = _create_file_list_items(file_list, clicked_file)
        # Save the clicked file as the last opened file
        return clicked_file, file_items, clicked_file
    
    # Callback to reload file list (initial load and refresh)
    @callback(
        [Output("results-file-list", "children", allow_duplicate=True),
         Output("results-file-list-store", "data")],
        [Input("results-create-results", "n_clicks"),
         Input("results-upload-results", "contents")],
        [State("results-result-files", "data"),
         State("last-opened-file-store", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def reload_file_list(create_clicks, upload_contents, current_selected_file, last_opened_file):
        # Reload file list when new file is created or uploaded
        model.available_results = model.get_available_results()
        file_list = model.available_results
        
        # Use current_selected_file if available, otherwise try to use last_opened_file
        file_to_select = current_selected_file
        if not file_to_select and last_opened_file and last_opened_file in file_list:
            file_to_select = last_opened_file
        
        file_items = _create_file_list_items(file_list, file_to_select)
        return file_items, file_list
    
    @callback(
        [Output("results-table", "children"),
         Output("last-opened-file-store", "data", allow_duplicate=True)],
        [Input("results-result-files", "data"),
         Input("results-upload-results", "contents"),
         Input("display-columns", "value")],
        [State("last-opened-file-store", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def update_results_table(results_result_file, results_upload_contents, display_columns, last_opened_file):
        # Update last opened file when a file is selected (not for uploads)
        if results_result_file and results_result_file != last_opened_file:
            last_opened_file = results_result_file
        
        if results_result_file or results_upload_contents:
            handle_result_file_load(results_result_file, results_upload_contents, model)
        else:
            return html.Div("No results available"), last_opened_file
    
        # If no results loaded, return empty message
        if not model.results or len(model.results) == 0:
            return html.Div("No results available"), last_opened_file
    
        if not display_columns:
            return html.Div("No columns selected"), last_opened_file
        else:
            df = pd.DataFrame(model.results)
            return create_results_table(df, select_all=True, display_columns=display_columns), last_opened_file
    
    
    @callback(
        [Output("display-columns", "options"),
         Output("display-columns", "value")],
        [Input("results-result-files", "data"),
         Input("results-upload-results", "contents")],
        [State("display-columns", "value"),
         State("file-configs-store", "data")]
    )
    def update_display_columns_options(results_result_file, results_upload_contents, current_columns, file_configs):
        # Determine the current file name
        current_file = results_result_file
        if not current_file and results_upload_contents:
            # For uploaded files, use a special key
            current_file = "__uploaded__"
        
        # Try to load results
        if not model.results:
            if results_result_file or results_upload_contents:
                handle_result_file_load(results_result_file, results_upload_contents, model)
        
        # If no results after loading, return empty
        if not model.results or len(model.results) == 0:
            # Clear columns when no results
            if current_file and file_configs and current_file in file_configs:
                # Keep the config but return empty for now
                return [], []
            return [], []
        
        columns = get_results_columns(model)
        options = [{"label": col, "value": col} for col in columns]
        
        # Load saved columns for this file from file_configs
        if current_file and file_configs and current_file in file_configs:
            saved_columns = file_configs[current_file].get('display_columns', [])
            # Filter to only include valid columns
            valid_columns = [col for col in saved_columns if col in columns]
            if valid_columns:
                return options, valid_columns
        
        # If no saved config, use current columns if valid
        if current_columns:
            valid_columns = [col for col in current_columns if col in columns]
            if valid_columns:
                return options, valid_columns
        
        # If no valid columns, return all columns
        return options, list(columns)

    @callback(
        Output("file-configs-store", "data"),
        [Input("display-columns", "value")],
        [State("results-result-files", "data"),
         State("results-upload-results", "contents"),
         State("file-configs-store", "data")],
        prevent_initial_call=True
    )
    def store_display_columns(selected_columns, results_result_file, results_upload_contents, file_configs):
        if file_configs is None:
            file_configs = {}
        
        # Determine the current file name
        current_file = results_result_file
        if not current_file and results_upload_contents:
            current_file = "__uploaded__"
        
        if current_file:
            if current_file not in file_configs:
                file_configs[current_file] = {}
            file_configs[current_file]['display_columns'] = selected_columns or []
            return file_configs
        
        return no_update
    
    @callback(
        [Output("results-message", "children"),
         Output("results-table", "children", allow_duplicate=True)],
        [Input("save-changes-button", "n_clicks")],
        [State("display-columns", "value")],
        prevent_initial_call=True
    )
    def save_results_changes(n_clicks, display_columns):
        if n_clicks:
            message, table = save_changes(model, display_columns)
            return message, table
        return no_update, no_update
    
    @callback(
        Output({"type": "row-select", "index": ALL}, "value"),
        [Input("select-all-button", "n_clicks"),
         Input("clear-all-button", "n_clicks")],
        [State({"type": "row-select", "index": ALL}, "id")]
    )
    def select_all_rows(select_clicks, clear_clicks, row_ids):
        ctx = callback_context
        if not ctx.triggered:
            return [no_update] * len(row_ids)
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "select-all-button":
            return [["selected"] for _ in row_ids]
        elif trigger_id == "clear-all-button":
            return [[] for _ in row_ids]
            
        return [no_update] * len(row_ids)
    
    @callback(
        Output("selected-results", "data"),
        [Input({"type": "row-select", "index": ALL}, "value")],
        [State("results-table", "children")]
    )
    def update_selected_results(selected_values, results_table):
        if not results_table or isinstance(results_table, str):
            return []
            
        selected_indices = []
        for i, value in enumerate(selected_values):
            if value and "selected" in value:
                selected_indices.append(i)
                
        return selected_indices
    

    @callback(
        Output("results-table", "children", allow_duplicate=True),
        [Input({"type": "delete-row", "index": ALL}, "n_clicks")],
        [State("display-columns", "value"),
         State({"type": "delete-row", "index": ALL}, "id")],
        prevent_initial_call=True
    )
    def handle_delete_row(delete_clicks, display_columns, delete_ids):
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update
            
        # Get the index of the row to delete
        trigger_id = ctx.triggered[0]['prop_id']
        if not trigger_id or "delete-row" not in trigger_id:
            return no_update
            
        # Find the index of the clicked button
        clicked_index = None
        for i, (click, delete_id) in enumerate(zip(delete_clicks, delete_ids)):
            if click and click > 0:  # Only consider actual clicks
                clicked_index = delete_id['index']
                break
                
        if clicked_index is not None:
            # Delete the row
            model.results.pop(clicked_index)
            
            # Update the table with remaining rows
            if model.results:
                df = pd.DataFrame(model.results)
                return create_results_table(df, select_all=True, display_columns=display_columns)
            else:
                return html.Div("No results available")
            
        return no_update

