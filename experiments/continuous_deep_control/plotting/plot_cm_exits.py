from parsers.plot_parser import *
from .plotter_classes import *
from .file_processing_classes import *
import numpy as np

def main(args=None):
    parser = CMPlotParser()
    args = parser.parse_args(args)
    file_processor = FileProcessing(args)
    manager = PlotManager(args)

    def input_file_patt_f(e, a):
        return re.compile("{e}_{a}_setting_(?P<setting>\d+)_run_(?P<run>\d+)_(?P<exit_type>BadExit|RightExit)\.txt".format(e=e, a=a))

    def get_plot_id_f(obj):
        if obj.args.separate_agent_plots:
            return "_".join([obj.agent_name, obj.args.env_name, obj.exit_type])
        else:
            return "_".join([obj.args.env_name, obj.exit_type])

    input_regex_groups = ["setting", "run", "exit_type"]

    for plot_id in file_processor.iterate_input_files(input_file_patt_f, input_regex_groups, get_plot_id_f):
        data = np.loadtxt(file_processor.full_fname, delimiter=',')
        manager.add(plot_id, file_processor.agent_name, file_processor.ag_params, file_processor.setting, data)

    synchronize_yaxis_options = {
        "mode": "y_idx",
        "sync_idx": file_processor.sync_idx,
        "keep_ymin": True,
    }
    manager.plot_and_save_all(synchronize_yaxis_options)

if __name__ == "__main__":
    main()