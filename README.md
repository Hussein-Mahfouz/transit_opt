# Transit Network Optimisation

![PyPI version](https://img.shields.io/pypi/v/transit_opt.svg)
[![Documentation Status](https://readthedocs.org/projects/transit_opt/badge/?version=latest)](https://transit_opt.readthedocs.io/en/latest/?version=latest)

This repo is meant to host code for the joint design of public transport schedules and DRT fleet sizes.
- For an overview of the research area, check [Planning, operation, and control of bus transport systems: A literature review](https://www.sciencedirect.com/science/article/abs/pii/S0191261515000454)
- The specific focus of this research is frequency setting problem.
- We aim to add DRT fleet sizing to the frequency setting problem. A relevant paper that describes the research area is [Joint design of multimodal transit networks and shared  autonomous mobility fleets](https://www.sciencedirect.com/science/article/pii/S235214651930016X)

> [!WARNING]  
> This is a WIP research project. You can find a rough overview of functionality in the notebooks folder, but the codebase will change in the next few weeks

## Features

- [ ]  Read GTFS and convert to matrix for optimisation
- [ ] Headway optimisation of existing bus network
    - [ ] Different objective functions
    - [ ] Configurable constraints
    - [ ] Algorithms: PSO
- [ ] Adding DRT fleet sizing to problem
- [ ] Writing output back to GTFS


## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
