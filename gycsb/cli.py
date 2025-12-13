import argparse
from gycsb.visualizer import create_app as visualizer_app
from gycsb.obsidian_server import create_app as obsidian_server_app
from gycsb.Runner import run_single_benchmark, run_all_benchmarks
from gycsb.ConfigLoader import __load_config
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='gYCSB Benchmark CLI')
    subparsers = parser.add_subparsers(dest='mode', help='modes of gYCSB')
    
    
    # single run mode
    single_run = subparsers.add_parser('singlerun', help='Single run mode')
    single_run.add_argument('--runner_config', type=str, required=True, help='The config file path for the runner')
    single_run.add_argument('--running_name', type=str, required=True, help='The running name for the benchmark. This will be the name of the results file in the results folder.')
    
    
    # batch run mode
    batch_run = subparsers.add_parser('batchrun', help='Batch run mode')
    batch_run.add_argument('--runner_config', type=str, required=True, help='The config file path for the runner')
    batch_run.add_argument('--running_name', type=str, required=True, help='The running name for the benchmark. This will be the name of the results file in the results folder.')
    batch_run.add_argument('--variables', type=list, required=True, help='The variables to iterate over for batch running. Must be provided.')
    
    # visualizer mode
    visualizer = subparsers.add_parser('visualizer', help='Start the visualizer server')
    visualizer.add_argument('--port', type=int, default=8052, help='The port to start the visualizer server')
    visualizer.add_argument('--host', type=str, default='0.0.0.0', help='The host to start the visualizer server')
    
    # obsidian server mode
    obsidian_server = subparsers.add_parser('obsidian_server', help='Start the obsidian server')
    obsidian_server.add_argument('--port', type=int, default=8051, help='The port to start the obsidian server')
    obsidian_server.add_argument('--host', type=str, default='0.0.0.0', help='The host to start the obsidian server')
    
    args = parser.parse_args()
    if args.mode == 'visualizer':
        app = visualizer_app()
        app.run(host=args.host, debug=True, port=args.port)
    elif args.mode == 'obsidian_server':
        app = obsidian_server_app()
        app.run(host=args.host, debug=True, port=args.port)
    elif args.mode == 'singlerun':
        path = Path(args.runner_config)
        config = __load_config(path)
        run_single_benchmark(config, args.running_name)
    elif args.mode == 'batchrun':
        path = Path(args.runner_config)
        config = __load_config(path)
        run_all_benchmarks(config, args.running_name, args.variables)
    else:
        parser.print_help()


        
        
        
if __name__ == "__main__":
    main()