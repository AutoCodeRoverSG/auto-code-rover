# Testing

This project is configured with CI workflows to execute the testing suite on every PR and push to the `main` branch, as well as pushes to the `pytest-ci` branch. The testing suite is also configured to run locally using the `tox` tool.

## Setup

To begin running the tests locally, it is assumed that the `auto-code-rover` environment has already been setup. Refer to the [README.md](README.md) for instructions on how to setup the environment.

The testing suite uses the following libraries and tools:
- Tox, to configure the tests
- Pytest, to execute the tests
- Coverage, (the Coverage.py tool) to measure the code coverage

In the `auto-code-rover` environment, install the required libraries by running the following command:

```bash
conda install -y tox
```

and execute the tox commands (configured in `tox.ini`) to run the tests:

```bash
tox -e py
```

The test results and the test coverage report will be displayed in the terminal, with a `coverage.xml` file in the Cobertura format generated in the project's root directory.