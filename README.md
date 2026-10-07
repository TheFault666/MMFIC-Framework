<p align="center">
  <img src="/static/logo.png" alt="MMFIC Logo" width="500"/>
</p>
# MMFIC: Multi-Modal Forum Identity Correlation Framework 

*Currently in Public Beta. See the Wiki for known issues and contribution guidelines.*

**MMFIC** (Multi-Modal Forum Identity Correlation) is a Open Source Intelligence (OSINT) framework designed to track, correlate, and unmask anonymous threat actors across heavily fortified Darkweb forums (.onion) and Clearnet platforms (like X/Twitter). 

The framework is divided into two primary subsystems:
1. **Trowler (Data Acquisition Engine):** A LLM-powered headless crawler built to bypass EndGame, DDoS-Guard, and strict login walls.
2. **The Similarity Engine:** A Machine Learning pipeline that extracts stylistic, linguistic, and behavioral metadata from crawled posts to correlate user pseudonyms across different platforms.

## Key Features

### Trowler (Crawling Subsystem)
* **Anti-Bot Circumvention:** Utilizes Playwright with Stealth plugins and a custom Human-In-The-Loop (HITL) fallback to bypass CAPTCHAs and Tor wait queues.
* **Tor Circuit Rotation:** Dynamically requests fresh Tor identities and exit nodes to prevent IP bans.
* **LLM-Powered Parsing:** No more hardcoded BeautifulSoup scrapers. Trowler uses LiteLLM (Gemini, GPT, Claude, or local Ollama) to autonomously infer the structure of unknown forums and extract JSON data.

### MMFIC Core (Identity Subsystem)
* **Linguistic Fingerprinting:** Generates vectorized profiles of users based on their grammar, syntax, and posting behaviors.
* **Cross-Forum Correlation:** Allows analysts to input a target user and mathematically score their similarity against thousands of other pseudonyms in the database to uncover alternate identities.
* **Hybrid Interface:** Run targeted operations headless via the CLI (`main.py`) or manage long-running data acquisition via the web UI (`dashboard.py`).

## Installation

### 1. Prerequisites
* **Python 3.10+**
* **Tor Expert Bundle** or **Tor Browser** (Running locally with SOCKS5 on `127.0.0.1:9150` or `9050`. Control port must be exposed).

### 2. Setup
```bash
git clone https://github.com/yourusername/MMFIC-Framework.git
cd MMFIC-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright Chromium binaries
playwright install chromium
```

### 3. Configuration
Copy the environment template and add your preferred LLM API key:
```bash
cp .env.example .env
```
Edit `.env` and add your API keys. Gemini is used by default, but any LiteLLM-compatible provider works.

## Usage

**CLI Mode (Fast & Targeted)**
```bash
python main.py
```
*Select Mode 1/2 for the Trowler Crawler, or Mode 3 for the MMFIC Identity Engine.*

**Dashboard Mode (Long-running Jobs)**
```bash
python dashboard.py
```
*Access the web UI at `http://127.0.0.1:5000` to manage Trowler crawls, view logs, and handle CAPTCHAs interactively.*

## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. 
Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
