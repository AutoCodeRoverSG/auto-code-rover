# Testing

This project is configured with CI workflows to execute the testing suite on every PR and push to the `main` branch, as well as pushes to the `pytest-ci` branch. The testing suite is also configured to run locally using the `tox` tool.

## Setup

To begin running the tests locally, it is assumed that the `auto-code-rover` environment has already been setup. Refer to the [README.md](README.md) for instructions on how to setup the environment.

The testing suite uses the following libraries and tools:
- Tox, to configure the tests
- Pytest, to execute the tests
- Coverage, (the Coverage.py tool) to measure the code coverage


Creating the `auto-code-rover` environment using the `environment.yaml` file:
```bash
conda env create -f environment.yml
conda activate auto-code-rover
```


In the `auto-code-rover` environment, add `conda-forge` as a channel, then install the required libraries by running the following command:

```bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
conda install -y tox
```

## Running the tests locally

To run the tests, execute the `tox` command (you can view the configurations in `tox.ini`) to run the tests:

```bash
tox
```

Alternatively, command to activate the `auto-code-rover` environment and run the tests:

```bash
conda activate auto-code-rover && tox
```

The test results and the test coverage report will be displayed in the terminal, with a `coverage.xml` file in the Cobertura format generated in the project's root directory.

## Modifying the `tox.ini` file

To enable missing statement coverage, add the following section the `tox.ini` file:

```ini
[coverage:report]
show_missing = True
```
