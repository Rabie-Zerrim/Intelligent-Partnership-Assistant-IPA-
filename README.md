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

1️⃣ Data Scraping

🌐 Selenium scrapes company names, descriptions, ratings, and reviews.

💾 Data is stored in a MySQL database for further processing.

2️⃣ Data Cleaning & Preprocessing

🧹 Removes duplicates and irrelevant information.

🔑 Extracts IT-related keywords to calculate a keyword_score.

📊 Normalizes ratings & review counts with StandardScaler.

3️⃣ Fuzzy Matching

🔎 Uses FuzzyWuzzy to match company names.

✅ Handles typos, spelling variations, and partial matches.

4️⃣ Model Training

🧠 Trains a Support Vector Classifier (SVC) using:

⭐ Ratings

📝 Review count

🔑 Keyword score

📈 Produces an eligibility prediction with probability scores (via sigmoid).

**Outputs eligibility with probability scores (sigmoid).

**Bot Capabilities

✅ Check company eligibility

📊 Compare two companies with a graph

🏆 List all eligible companies

🔄 Reset predictions & view stats

💾 Save chat to CSV / 🗑 Delete history

<img width="829" height="948" alt="Screenshot 2025-05-12 232512" src="https://github.com/user-attachments/assets/64912912-c86a-4196-a124-bd398873e53f" />
<img width="809" height="679" alt="Screenshot 2025-05-12 232549" src="https://github.com/user-attachments/assets/e6b9a0e5-12e2-400d-97d1-b4e008a14204" />
<img width="817" height="926" alt="Screenshot 2025-05-12 233009" src="https://github.com/user-attachments/assets/a2f15d1a-8dd4-4169-8a4b-ab2fa518d17f" />
<img width="827" height="955" alt="Screenshot 2025-05-12 233607" src="https://github.com/user-attachments/assets/d7179f2f-50d8-4248-b6ed-7f67c4391d5b" />

API Endpoints

POST /predict → Get eligibility predictions

POST /ask → Ask queries in natural language

POST /compare_companies_graph → Compare two companies (graph output)

POST /save_chat → Save chat as CSV

DELETE /delete_chat → Clear chat history

🚀 Setup & Installation

1️⃣ Install Dependencies
pip install flask flask_sqlalchemy flask_cors joblib spacy matplotlib mysql-connector-python fuzzywuzzy[speedup]


Download SpaCy model:

python -m spacy download en_core_web_sm

2️⃣ Database Configuration

In your Flask app (flask_bot.py or app.py), update the database URI:

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://<username>:<password>@localhost:3306/<database_name>'

3️⃣ Run the Flask App

Start the server with:

python flask_bot.py

4️⃣ Access the API

You can interact with the API using:

🛠️ Postman

💻 cURL

🌐 Any frontend integration (Angular/React/etc.)

📌 Tech Stack
Layer	Tools / Libraries
Backend	Flask, SQLAlchemy
Database	MySQL
ML	scikit-learn (SVC, StandardScaler)
NLP	SpaCy, TF-IDF
Scraping	Selenium
Matching	FuzzyWuzzy
Visualization	Matplotlib
