# Overview

The files are divided as:
- `config.py`: file with configuration classes corresponding to the plots (e.g. line colors, font sizes, tick formatters...)
- `file_processing_classes.py`: contains auxiliary classes used in the main code to process the files in the results directory
- `plotter_classes.py`: classes used to process the data read from the files and plot them

The remaining files are all similar and are used to generate the plots in the paper.

# Commands to generate plots from the paper
For additional arguments not appearing in the examples below, see `parsers/plot_parser.py`
## Main benchmark plots:

### Grouped by specific temperatures
````
pipenv run python3 -m plotting.plot_benchmarks <env_name> --divide_type entropy_scale --how_to_group top20 --config_class BenchmarksPlotConfig --separate_agent_plots
````

### Grouped by high x low temperatures

- Curve plots:
````
pipenv run python3 -m plotting.plot_all_high_all_low <env_name>  --divide_type all_high_all_low --how_to_group top20 --config_class BenchmarksPlotConfig
````
- Bar plots:
````
pipenv run python3 -m plotting.plot_all_high_all_low_bar <env_name>  --divide_type all_high_all_low --how_to_group top20 --config_class BenchmarksBarPlotConfig --bar --normalize
````

## Sensitivity plots
`<desired_hyper>` was used as `qf_vf_lr` or `pi_lr` 
````
pipenv run python3 -m plotting.plot_hyperparameter_sensitivity <env_name> --divide_type entropy_scale --how_to_group top20 --config_class HyperSensPlotConfig --separate_agent_plots --hyperparam_for_sensitivity <desired_hyper>
````

## Gridworld plots
````
pipenv run python3 -m plotting.plot_cm_exits <env_name> --separate_agent_plots --how_to_group best --config_class CMPlotConfig --agent_names ForwardKL ReverseKL GMMForwardKL GMMReverseKL
````