# Service Me

Service Me is a Chrome extension for ticket entry and automation in ServiceNow. It uses a classical (non-LLM) model, TF-IDF with logistic regression plus cosine-similarity retrieval, to speed up filling out contacts. The model runs entirely in the browser.

> [!TIP]
> Suggestions and inputs from the extension should always be checked by the user for accuracy.

> [!WARNING]
> Service Me has no function straight out of this repo. The training data is not included for security reasons, so you must train and export your own model first (see [Setup](#setup)).

## Features

Given a short description, Service Me shows:

- **Inquiry or incident:** a guess at whether the ticket is a General Inquiry or an Incident.
- **Time estimate:** "AT THE WINDOW" for inquiries. For incidents, the median and range of similar past incidents.
- **Priority, category and record source:** shown for reference only. They are not filled in.
- **Similar tickets:** the three closest past tickets.
- **Suggested short description:** the most typical wording among close past tickets, with no names, EIDs, emails or phone numbers. It only appears when enough close tickets exist.

An **Auto-fill** button then opens the new contact page in a new tab and fills in:

| Field | Value |
| --- | --- |
| Requested For | the requester EID typed in the popup (Requested By follows automatically) |
| Location | from Settings |
| Assignment Group | from Settings |
| Assigned To | the EID from Settings |
| Contact Method | always `Walk-In` |
| Ticket Type | `General Inquiry` or `Incident`, from the model's guess |
| Short description | the text typed in the popup |

Service Me **never saves or submits** a ticket. You review the form and save it yourself.

## How it works

- **Features:** TF-IDF on the short description (word 1- and 2-grams, sublinear term frequency).
- **Classification:** logistic regression for inquiry vs. incident and for priority, category and record source. A field where one value covers at least 95% of tickets uses that value as a fixed default.
- **Retrieval:** cosine similarity over past tickets gives the similar tickets, the time estimate and the suggested wording.
- **Export:** the trained model is written to `extension/model/model.json` and run in JavaScript by `engine.js`. It matches the Python model.

## Setup

Requirements: Python 3 with `scikit-learn`, `pandas`, `numpy` and `joblib`, and Chrome.

1. **Add your data.** Put your ServiceNow ticket export at `data/raw/raw.csv`. Column names are set in `config.py`.
2. **Split and train.**
   ```
   python src/split.py
   python src/train.py
   python src/evaluate.py
   ```
   `evaluate.py` scores the model on held-out tickets and prints counts and rates only, never ticket text.
3. **Train the final model and export it.**
   ```
   python src/train.py --final
   python src/export.py
   ```
4. **Load the extension.** Open `chrome://extensions`, turn on Developer mode, choose **Load unpacked** and select the `extension` folder.
5. **Configure it.** Open the extension popup, then Settings, and enter the contact page URL, Location, Assignment Group and your Assigned To EID.

## Usage

1. Click the extension on any page.
2. Enter the short description and the requester's EID.
3. Review the guesses, then click **Auto-fill**.
4. Check the filled form and save it yourself.

## Privacy and security

- `extension/model/model.json` contains past ticket text. **Keep it out of git** and out of any shared copy.
- The requester EID is used only for auto-fill and is never stored.
- Settings are stored in Chrome (`chrome.storage.local`), not in the repo. The unsent short description draft is held only for the browser session.

## Plans

- [ ] Adjust and suggest short descriptions to conform to the standardized description schema.
- [ ] Automate descriptions depending on the contact method, and broaden the training data.
- [ ] Improve similar ticket matching.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config.py` | Thresholds, column names and paths |
| `load.py`, `features.py`, `model.py` | Data loading, TF-IDF features and the model |
| `split.py`, `train.py`, `evaluate.py` | Split, train and score |
| `export.py` | Writes `extension/model/model.json` |
| `readiness_tests.ipynb` | Readiness checks for the model |
| `extension/` | The Chrome extension (Manifest V3) |