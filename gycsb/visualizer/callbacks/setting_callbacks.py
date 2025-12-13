from dash import callback, Output, Input, State
from gycsb.ConfigLoader import load_system_setting, save_system_setting


def register_setting_callbacks():
    """Register callbacks for the setting page"""
    
    @callback(
        Output("config-path-display", "children"),
        Input("operation-file-dir", "value")
    )
    def update_config_display(operation_file_dir):
        """Update the configuration path display"""
        if operation_file_dir:
            return operation_file_dir
        return "Not configured"
    
    @callback(
        Output("operation-file-dir", "value"),
        Input("operation-file-dir", "value"),
        State("operation-file-dir", "value"),
        prevent_initial_call=True
    )
    def save_operation_file_dir(new_value, state_value):
        """Save the operation file directory to config when it changes"""
        if new_value and new_value != state_value:
            try:
                config = load_system_setting()
                config['operation_file_dir'] = new_value
                save_system_setting(config)
            except Exception as e:
                print(f"Error saving config: {e}")
        return new_value

