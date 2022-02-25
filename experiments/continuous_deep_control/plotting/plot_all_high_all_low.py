from parsers.plot_parser import *
from .plotter_classes import *
from .file_processing_classes import *

def main(args=None):
    parser = BenchmarksPlotParser()
    args = parser.parse_args(args)
    file_processor = FileProcessing(args)
    manager = PlotManager(args)

    def input_file_patt_f(e, a):
        return re.compile("{e}_{a}_setting_(?P<setting>\d+)_run_(?P<run>\d+)_EpisodeRewardsLC\.txt".format(e=e, a=a))

    def get_plot_id_f(obj):
        return obj.args.env_name

    input_regex_groups = ["setting", "run"]

    for plot_id in file_processor.iterate_input_files(input_file_patt_f, input_regex_groups, get_plot_id_f):
        if not manager.load_existing_data(plot_id):
            data = file_processor.load_and_unroll_current_file()
            manager.add(plot_id, file_processor.agent_name, file_processor.ag_params, file_processor.setting, data)

    synchronize_yaxis_options = {
        "mode": "from_file",
        "target_call_id": "_".join(["SplitAgents", args.how_to_group, "entropy_scale"]),
        "sync_idx": file_processor.sync_idx
    }

    manager.plot_and_save_all(synchronize_yaxis_options)
    manager.save_all_data() # To avoid processing again when plotting the next time

if __name__ == "__main__":
    main()
