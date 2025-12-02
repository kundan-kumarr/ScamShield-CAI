# ScamShield-CAI: Constitutional AI for Financial Scam Prevention

This repo implements a small Constitutional AI pipeline that aligns an open LLM
to **refuse** helping with financial scams, fraud, phishing, and exploitation,
while remaining helpful for benign finance questions (budgeting, savings, etc.).

The pipeline is:

1. **Red-teaming prompts** → generate unsafe responses from a base chat model.
2. **Self-critique + revision** guided by a **Financial Scam Constitution**.
3. Build:
   - an **SFT dataset** (prompt + safe revision),
   - a **preference dataset** (safe vs unsafe answers).
4. Train:
   - **SFT model** on helpful chat + CAI revisions.
   - **DPO model** on the preference pairs.
5. Evaluate safety & helpfulness with red-team & benign prompts.

## Quickstart

```bash
git clone <this-repo>
cd scamshield-cai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
