🌐 Intelligent Partnership Assistant (IPA)
📖 Overview

The Intelligent Partnership Assistant (IPA) is a Flask-based AI bot designed to streamline partnership evaluations using machine learning, NLP, web scraping, and fuzzy matching.
It provides company eligibility predictions, comparisons, and insights through a simple API and natural language queries.

✨ Key Features

🔮 Company Eligibility Prediction – Predicts partnership eligibility using a trained ML model (SVC).

💬 Natural Language Queries – Ask questions in plain English to check eligibility, compare companies, or get statistics.

⚖️ Company Comparison – Generates visual comparisons of companies (ratings, reviews, eligibility).

🌐 Automated Data Scraping – Collects company data (reviews, ratings, descriptions) via Selenium.

🧹 Data Cleaning & Preprocessing – Ensures quality input with deduplication, normalization, and keyword scoring.

🧠 Machine Learning Integration – Support Vector Classifier trained on structured company data.

🔎 Fuzzy Matching – Matches company names accurately even with typos (via FuzzyWuzzy).

💾 Chat History Management – Save chats as CSV, or delete history when needed.

⚙️ How It Works

Data Scraping

Selenium scrapes company names, descriptions, ratings, and reviews.

Data is stored in a MySQL database for processing.

Data Cleaning & Preprocessing

Removes duplicates and irrelevant info.

Extracts IT-related keywords to calculate a keyword_score.

Normalizes ratings & review counts with StandardScaler.

Fuzzy Matching

Uses FuzzyWuzzy for flexible company name matching.

Model Training

Trains a Support Vector Classifier (SVC) on features:

⭐ Ratings

📝 Review count

🔑 Keyword score

Outputs eligibility with probability scores (sigmoid).

Bot Capabilities

✅ Check company eligibility

📊 Compare two companies with a graph

🏆 List all eligible companies

🔄 Reset predictions & view stats

💾 Save chat to CSV / 🗑 Delete history

API Endpoints

POST /predict → Get eligibility predictions

POST /ask → Ask queries in natural language

POST /compare_companies_graph → Compare two companies (graph output)

POST /save_chat → Save chat as CSV

DELETE /delete_chat → Clear chat history

🚀 Setup & Installation
1. Install Dependencies
   pip install flask flask_sqlalchemy flask_cors joblib spacy matplotlib mysql-connector-python fuzzywuzzy[speedup]


Download SpaCy model:

python -m spacy download en_core_web_sm

2. Database Configuration

Update your Flask app:

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://<username>:<password>@localhost:3306/<database_name>'

3. Run the Flask App
   python flask_bot.py

4. Access the API

Use Postman, cURL, or integrate with a frontend (e.g., Angular/React) to query endpoints.

📌 Tech Stack

Backend: Flask, SQLAlchemy

Database: MySQL

ML: scikit-learn (SVC, StandardScaler)

NLP: SpaCy, TF-IDF

Scraping: Selenium

Matching: FuzzyWuzzy

Visualization: Matplotlib
