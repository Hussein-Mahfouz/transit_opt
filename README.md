# Transit Network Optimisation

![PyPI version](https://img.shields.io/pypi/v/transit_opt.svg)
[![Documentation Status](https://readthedocs.org/projects/transit_opt/badge/?version=latest)](https://transit_opt.readthedocs.io/en/latest/?version=latest)

This repository provides a framework for the joint optimization of public transport schedules and Demand Responsive Transit (DRT) fleet sizes.

For an overview of the research area, see [Planning, operation, and control of bus transport systems: A literature review](https://www.sciencedirect.com/science/article/abs/pii/S0191261515000454). The specific focus of this implementation is the frequency setting problem, extending it to multimodal networks (PT continuous frequency setting + DRT discrete fleet sizing). A relevant foundational paper is [Joint design of multimodal transit networks and shared autonomous mobility fleets](https://www.sciencedirect.com/science/article/pii/S235214651930016X).

## Features

- Parse and extract operational data directly from raw GTFS feeds.
- Headway optimization of existing transit networks using metaheuristics (PSO via PyMOO).
- Joint optimization of Fixed-Route PT headways and DRT fleet sizing.
- Pluggable framework for customized objective functions (e.g., waiting time, service coverage) and configurable constraints (e.g., fleet limits).
- Automated reconstruction of optimized solutions back into valid GTFS outputs for downstream simulation (e.g., MATSim).

## Installation

This project uses `uv` for dependency management. To set up the virtual environment:

```bash
uv venv --python 3.12 .transit_opt_uv
source .transit_opt_uv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

## Usage

The core functional pipeline runs via the CLI using a YAML configuration file. The configuration defines the input GTFS, the allowable headway parameters, optimization constraints, evaluation intervals, and output directories.

To run the optimization pipeline:

```bash
python scripts/run.py --config configs/config_template.yaml
```

If you are running iterative setups (such as starting an optimization run with seeded samples from a previous run), you can specify the target iteration index:

```bash
python scripts/run.py --config configs/iteration_01/your_config.yaml --iteration 1
```

### Notebooks

You can find conceptual walk-throughs, data visualisations, and implementation examples in the `notebooks/` directory.

> [!WARNING]
> While notebooks are a helpful starting point to understand the framework's mechanics, some may be outdated compared to the active `scripts/run.py` pipeline.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
