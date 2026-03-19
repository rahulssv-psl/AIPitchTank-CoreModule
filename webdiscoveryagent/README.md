## Web Discovery Agent

---

### Quick Start

Follow these steps to get the agent up and running on your local machine.

#### 1. Clone the Repository

First, grab the code and navigate into the project directory:

```bash
git clone https://github.com/amitkumar-ghatwal/PowerPortAI-Ecosystem.git
cd webdiscoveryagent

```

#### 2. Set Up the Environment

Use **uv** to manage the Python environment and dependencies.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install uv
python -m pip install -U pip uv
uv venv --clear
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies using uv sync
uv sync

```

Setup Credentials inside `.env`

```bash
cat <<EOF > .env
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_API_KEY=your_api_key_here
PROJECT_ID=your_project_id_here
EOF

```

#### 3. Run the Agent

Once the dependencies are synced, execute the LangChain agent implementation directly through `uv`:

```bash
uv run langchain_agent_impl.py

```

---

