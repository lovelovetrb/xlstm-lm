lint:
	rye run ruff check src
	rye run mypy src --config-file pyproject.toml --ignore-missing-imports --no-namespace-packages
	
format:
	rye run ruff format src
	rye run ruff check --fix --select I src
