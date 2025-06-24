# Phishing URL Detector Chrome Extension

## Overview

This Chrome extension helps detect potential phishing websites by analyzing the current page URL. It uses a two-step detection process:

1. **Visual Mimic Detection:**The extension first checks if the domain visually mimics any well-known legitimate domains (e.g., `g00gle.com` mimicking `google.com`). It normalizes the domain to catch common character substitutions (like `0` for `o`, `1` for `l`, etc.) and calculates similarity scores against a whitelist of trusted domains. If a suspicious mimicry is detected, the extension immediately flags the site as **PHISHING** without running further analysis.
2. **Machine Learning Model Prediction:**
   If no visual mimicry is found, the extension extracts various URL features (lengths, special characters, presence of suspicious keywords, etc.) and feeds them into a TensorFlow.js model. This model predicts the likelihood that the site is phishing based on trained patterns. The extension then displays a verdict: **SAFE** or **PHISHING** with a confidence percentage.

## How it works

- On page load, the content script:
  - Extracts and normalizes the current URL's domain.
  - Compares the domain to a list of known legitimate domains for visual similarity using a custom normalization and Levenshtein distance calculation.
  - If similarity exceeds a threshold, it immediately flags the site as phishing.
  - Otherwise, it loads a TensorFlow.js model and a scaler to process URL features and predict phishing probability.
  - Shows a fixed notification box on the page indicating the verdict.

## Features

- **Fast Visual Mimic Check:** Detects common phishing tactics that use lookalike domains with subtle character substitutions.
- **Machine Learning Model:** Uses a pre-trained neural network to analyze URL characteristics beyond simple matching.
- **User-friendly Alerts:** Displays clear warnings or safe indicators directly on the page.

## Known Legitimate Domains (Whitelist)

The extension currently compares domains against these trusted sites:

- pepper.co.il
- paypal.com
- google.com
- apple.com
- microsoft.com
- amazon.com
- bankofamerica.com
- isracard.co.il
- ebay.com
- facebook.com
- btl.gov.il

## Installation

1. Clone or download this repository.
2. Load the extension into Chrome via `chrome://extensions` > "Load unpacked" and select the project folder.

## License

This code is provided for academic demonstration only.
All rights to the software, model, and implementation are reserved by Maya Levi.
No part of this repository may be used, modified, reproduced, distributed, or integrated into any product or system
without explicit written permission.

© 2025 Maya Levi. All rights reserved.

For contact: mayalevi6363@gmail.com
