alias create_venv="uv venv --python 3.13"
alias activate_venv="source .venv/bin/activate"
alias pip="uv pip"
alias pipinstall="pip install -r requirements.txt"
alias pipfreeze="pip freeze > requirements.txt"

if [ -d ".venv" ]; then
    activate_venv
fi

# update python path
export PYTHONPATH="$(pwd)"

export ANTHROPIC_API_KEY="op://Personal/Claude/credential"

export MODEL_NAME="claude-3-5-haiku-20241022"

export DOCUMENT_PATH="documents/fcn.tex"
alias main="op run -- uv run main.py"
alias ui="op run -- streamlit run ui.py"