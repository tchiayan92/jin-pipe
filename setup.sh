curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --allow-existing
uv pip install -e ".[dev]"
uv pip install -e ".[vad,asr,filter,gpu,viewer]"
source .venv/bin/activate