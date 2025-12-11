import json
import time
import os
import glob
from visualizer.config_utils import load_config

class VisualizerModel:
    def __init__(self):
        self.results_dir = load_config()['results_folder']
        self.available_results = self.get_available_results()
        self.current_file = None
        self.results = None


    def _load_status(self):
        """Load benchmark status from file"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_status(self, status):
        """Save benchmark status to file"""
        if status is None:
            if os.path.exists(self.status_file):
                os.remove(self.status_file)
            return
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=4)
        self.current_status = status
        
        
        
    def get_available_results(self):
        """Get list of available result files"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            return []
            
        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        return [os.path.basename(f) for f in result_files]

    def load_results_from_file(self, file_path: str):
        """Load results from a specific file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Support new format with 'results' field, or old format (direct array)
                if isinstance(data, dict) and 'results' in data:
                    self.results = data['results']
                elif isinstance(data, list):
                    self.results = data
                else:
                    # If it's a dict without 'results', try to use it as is
                    self.results = data
            self.current_file = file_path
            return True
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return False
        
    def generate_new_results_file(self, filename: str):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        filepath = os.path.join(self.results_dir, filename)
        
        if os.path.exists(filepath):
            raise Exception(f"File {filepath} already exists")
        
        self.current_file = filepath
        
        with open(filepath, 'w') as f:
            json.dump([], f, indent=4)
        
        return True
            
    def check_current_file(self):
        if self.current_file:
            if os.path.exists(self.current_file):
                results = self.load_results_from_file(self.current_file)
                if results == True:
                    return True
        return False

    def _save_results(self, result):
        """Save results to a JSON file"""
        results = []
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        if self.current_file:
            filepath = self.current_file
            
            self.load_results_from_file(filepath)
            self.results = self.results + [result]
            
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            self.current_file = filepath
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        print(f"Saving results to {filepath}")
            
        self.available_results = self.get_available_results()

