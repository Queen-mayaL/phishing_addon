import pandas as pd
import re
import tldextract
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import json
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance
import ipaddress
import os

# === Setup ===
suspicious_keywords = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'banking', 'confirm', 'signin']

known_legit_domains = [
    'pepper.co.il', 'paypal.com', 'google.com', 'apple.com', 'microsoft.com',
    'amazon.com', 'bankofamerica.com', 'isracard.co.il', 'ebay.com', 'facebook.com', 'btl.gov.il'
]

def get_base_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def normalize(text):
    replacements = {'0': 'o', '1': 'l', '3': 'e', '5': 's', '7': 't', '@': 'a'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.lower()

def has_ip(url):
    try:
        ipaddress.ip_address(urlparse(url).netloc)
        return 1
    except ValueError:
        return 0

def visual_similarity(url):
    domain = normalize(get_base_domain(url))
    return max(fuzz.ratio(domain, normalize(get_base_domain(legit))) for legit in known_legit_domains)

def edit_distance_score(url):
    domain = normalize(get_base_domain(url))
    return max(1 - levenshtein_distance(domain, normalize(get_base_domain(legit))) / max(len(domain), 1)
               for legit in known_legit_domains)

def extract_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    base_domain = get_base_domain(url)
    domain = parsed.netloc

    visual_sim = visual_similarity(url)
    edit_sim = edit_distance_score(url)

    suspicious_sim = int((visual_sim > 80) and (edit_sim > 0.7))

    contains_legit_brand = int(any(
        legit_domain.split('.')[0] in base_domain and base_domain != legit_domain for legit_domain in known_legit_domains))

    features = {
        'url_length': len(url),
        'hostname_length': len(domain),
        'path_length': len(parsed.path + parsed.query),
        'num_special_chars': len(re.findall(r'[./?\-_@=]', url)),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_subdomains': len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'has_ip': has_ip(url),
        'uses_https': int(parsed.scheme == 'https'),
        'tld': ext.suffix if ext.suffix in ['com', 'net', 'org', 'ru', 'top', 'cn', 'edu', 'co.uk', 'ca'] else 'other',
        'contains_suspicious_keyword': int(any(word in url.lower() for word in suspicious_keywords)),
        'has_hyphen': int('-' in base_domain),
        'is_known_legit': int(base_domain in known_legit_domains),
        'visual_similarity': visual_sim,
        'edit_similarity': edit_sim,
        'suspicious_similarity_flag': suspicious_sim,
        'contains_legit_brand_substring': contains_legit_brand
    }

    return features

# === Load Dataset ===
if not os.path.exists('combined_dataset.csv'):
    raise FileNotFoundError("❌ 'combined_dataset.csv' not found. Please provide a dataset file with 'url' and 'label' columns.")

base_df = pd.read_csv('combined_dataset.csv')

# === Balance Dataset ===
df_majority = base_df[base_df.label == 0]
df_minority = base_df[base_df.label == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
balanced_df = pd.concat([df_majority, df_minority_upsampled])

# === Feature Extraction ===
features_df = balanced_df['url'].apply(extract_features).apply(pd.Series)
features_df = pd.get_dummies(features_df, columns=['tld'], drop_first=True)
features_df.to_csv("debug_features.csv")
final_df = pd.concat([features_df, balanced_df['label']], axis=1)

X = final_df.drop('label', axis=1)
y = final_df['label']

with open('feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Random Forest ===
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
joblib.dump(rf_model, 'rf_model2.pkl')

# Feature importance
importances = rf_model.feature_importances_
sorted_idx = importances.argsort()[::-1]
plt.figure(figsize=(10, 5))
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')

# === Neural Network ===
print("Training Neural Network...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler2.pkl')

scaler_params = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
with open('scaler_params2.json', 'w') as f:
    json.dump(scaler_params, f)

nn_model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
nn_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=callbacks)
loss, acc = nn_model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Accuracy: {acc:.4f}")
nn_model.save('phishing_model2.h5')

print("\n✅ All models trained and saved successfully!")