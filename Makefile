lint:
	ruff format src
	ruff check --fix --select I src