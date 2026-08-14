# SALLY - Science Artificial Learning Logic and You

Offline AI assistant built on Debian with llama.cpp

## Setup
```bash
git clone https://github.com/techma2004/SALLY.git
cd SALLY
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp.env.example.env
# Edit.env with your keys
python main.py
