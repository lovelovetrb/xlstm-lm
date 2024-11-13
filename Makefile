lint:
	ruff check src
	mypy src --config-file pyproject.toml --ignore-missing-imports --no-namespace-packages

	
format:
	ruff format src
	ruff check --fix --select I src
