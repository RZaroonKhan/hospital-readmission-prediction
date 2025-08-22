# ----- config -----
PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python
STREAMLIT=$(VENV)/bin/streamlit

# On Windows, uncomment and adjust these (and comment out the POSIX ones above):
# PIP=$(VENV)/Scripts/pip.exe
# PYBIN=$(VENV)/Scripts/python.exe
# STREAMLIT=$(VENV)/Scripts/streamlit.exe

.PHONY: help venv install app lint fmt clean models freeze

help:
	@echo "make venv     - create virtual environment"
	@echo "make install  - install project dependencies"
	@echo "make app      - run Streamlit app"
	@echo "make lint     - run black & flake8 checks"
	@echo "make fmt      - auto-format with black"
	@echo "make models   - download model artifacts (placeholder)"
	@echo "make freeze   - export exact versions to requirements.txt"
	@echo "make clean    - remove venv and caches"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PYBIN) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

app:
	$(STREAMLIT) run app/app.py

lint:
	$(PIP) install black flake8
	$(VENV)/bin/black --check .
	$(VENV)/bin/flake8 .

fmt:
	$(PIP) install black
	$(VENV)/bin/black .

# Download large models (edit with your real URL or command; keeps repo small)
models:
	@echo ">> Place your download command here (curl/wget/gh release). Example:"
	@echo "   curl -L 'https://drive.google.com/yourfile' -o notebooks/artifacts/final_calibrated_rf.joblib"
	@echo "   # and ensure notebooks/artifacts/deployment_config.json is present"

# Write out exact versions from your current venv (useful after successful setup)
freeze:
	$(PIP) freeze > requirements.txt

clean:
	rm -rf $(VENV) **/__pycache__ .ipynb_checkpoints
