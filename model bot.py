from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from fuzzywuzzy import fuzz, process
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import random
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Optional
from flask_cors import CORS

# Setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost:3306/PartnershipPI'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)  # This will enable CORS for all routes

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ML model and scaler
scaler = joblib.load('scaler2.pkl')
model = joblib.load('svc_modelfinal2.pkl')

# Constants
IT_KEYWORDS = [
    "IT", "cloud", "spring boot", "microservices", "Java", "DevOps",
    "AI", "ML", "SaaS", "Big Data", "software", "training", "enterprise"
]
FEATURE_NAMES = ["ratings", "review_count", "keyword_score"]
MIN_MATCH_SCORE = 70
used_suggestions = set()

# Follow-up suggestions
follow_up_suggestions = [
    "Want to check another company's eligibility?",
    "Curious about the top 5 potential partners?",
    "Need to reset the predictions?",
    "Would you like to explore eligible companies?",
    "Interested in seeing who's not eligible?"
]

# TF-IDF pattern matching
patterns = {
    "compare_two_companies": [
        "compare (.+) and (.+)",
        "which is better for partnership: (.+) or (.+)",
        "who is more eligible between (.+) and (.+)",
        "is (.+) better than (.+)",
        "partnership comparison between (.+) and (.+)"
    ],
    "show_statistics": [
        "show me statistics", "eligibility summary", "give me some stats",
        "overall eligibility status", "how many companies are eligible",
        "eligibility breakdown", "summarize the eligibility"
    ],
    "check_specific_eligibility": [
        "is (.+) eligible", "is (.+) a potential partner",
        "check eligibility for (.+)", "does (.+) qualify",
        "can we partner with (.+)", "would (.+) be a good partner",
        "is (.+) suitable", "evaluate (.+) for partnership",
        "check if (.+) is eligible", "determine if (.+) is a good fit",
        "tell me if (.+) is eligible", "is (.+) qualified"
    ],
    "list_eligible_companies": [
        "list companies that are eligible", "show me eligible companies",
        "which companies are eligible", "top eligible companies",
        "eligible companies list", "show eligible companies"
    ],
    "reset_predictions": [
        "clear all predictions", "reset eligibility predictions",
        "start over with predictions", "remove all prediction data",
        "reset data", "clear predictions"
    ]
}
vectorizer = TfidfVectorizer().fit(sum(patterns.values(), []))

# Database model
class ScrapedCompany(db.Model):
    __tablename__ = 'scraped_company'
    id = db.Column(db.Integer, primary_key=True)
    contact = db.Column(db.String(255))
    description = db.Column(db.String(500))
    elegibility_precentage = db.Column(db.Float)
    elegible = db.Column(db.Boolean, default=False)
    keywords = db.Column(db.String(500))
    link = db.Column(db.String(255))
    reviews = db.Column(db.Integer)
    score = db.Column(db.Float)
    title = db.Column(db.String(255), nullable=False)

# Helpers
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["keyword_score"] = df["keywords"].apply(
        lambda x: len(set(map(str.strip, x.lower().split(","))) & 
                      set(k.lower() for k in IT_KEYWORDS))
    )
    return df

def clean_company_name(name: str) -> str:
    return re.sub(r'[^\w\s-]', '', name).strip().lower()

def find_company_match(company_name: str) -> Optional[ScrapedCompany]:
    clean_name = clean_company_name(company_name)
    company = ScrapedCompany.query.filter(func.lower(ScrapedCompany.title) == clean_name).first()

    if company:
        return company

    company = ScrapedCompany.query.filter(func.lower(ScrapedCompany.title).contains(clean_name)).first()
    if company:
        return company

    all_companies = ScrapedCompany.query.all()
    if not all_companies:
        return None

    matches = process.extract(
        clean_name, [c.title for c in all_companies],
        scorer=fuzz.token_sort_ratio, limit=3
    )

    if matches and matches[0][1] >= MIN_MATCH_SCORE:
        best_match = matches[0][0]
        return next((c for c in all_companies if c.title == best_match), None)

    return None

def calculate_eligibility(data: dict) -> dict:
    input_df = pd.DataFrame({k: [data[k]] for k in FEATURE_NAMES if k != "keyword_score"})
    input_df["keywords"] = [data["keywords"]]
    input_df = preprocess_data(input_df)
    input_df = input_df.drop(columns=["keywords"])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = expit(model.decision_function(input_scaled)[0])

    return {
        'elegible': bool(prediction),
        'percentage': f"{round(probability * 100, 2)}%",
        'probability': float(probability)
    }

def get_new_suggestion() -> str:
    global used_suggestions
    unused = list(set(follow_up_suggestions) - used_suggestions)
    if not unused:
        used_suggestions.clear()
        unused = follow_up_suggestions.copy()
    suggestion = random.choice(unused)
    used_suggestions.add(suggestion)
    return suggestion

def generate_comparison_graph(company1, company2, result1, result2):
    labels = ['Rating', 'Review Count', 'Eligibility Percentage']
    company1_data = [company1.score, company1.reviews, result1['probability'] * 100]
    company2_data = [company2.score, company2.reviews, result2['probability'] * 100]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], company1_data, width, label=company1.title)
    ax.bar([i + width/2 for i in x], company2_data, width, label=company2.title)

    ax.set_ylabel('Scores')
    ax.set_title('Company Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    graph_url = base64.b64encode(img.getvalue()).decode('utf-8')

    plt.close(fig)

    return graph_url

# API Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ["ratings", "review_count", "keywords"]
        missing = [f for f in required_fields if f not in data]

        if missing:
            return jsonify({
                'error': 'Oops! You missed something.',
                'missing': missing,
                'example': {
                    'ratings': 4.5,
                    'review_count': 50000,
                    'keywords': 'IT, Cloud, Spring Boot'
                }
            }), 400

        result = calculate_eligibility(data)

        return jsonify({
            'elegible': result['elegible'],
            'percentage': result['percentage'],
            'probability': result['probability']
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Something went sideways: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question', '').strip().lower()
        if not question:
            return jsonify({"answer": "Hmm, I need a question to work with! Try asking about a company's eligibility."})

        for pattern in patterns["check_specific_eligibility"]:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
                return check_company_eligibility(company_name)

        for pattern in patterns["compare_two_companies"]:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                name1 = match.group(1).strip()
                name2 = match.group(2).strip()
                return compare_companies(name1, name2)

        vectorized_question = vectorizer.transform([question])
        best_match, best_score = None, 0

        for intent, phrases in patterns.items():
            for phrase in phrases:
                vectorized_phrase = vectorizer.transform([phrase])
                similarity = cosine_similarity(vectorized_question, vectorized_phrase)[0][0]
                if similarity > best_score:
                    best_score = similarity
                    best_match = intent
        
        if best_match == "list_eligible_companies":
            return list_eligible_companies()
        elif best_match == "reset_predictions":
            return reset_predictions()
        elif best_match == "show_statistics":
            return show_statistics()

        return jsonify({
            "answer": "Hmm, that's outside my realm for now let me train a bit more",
            "suggestion": get_new_suggestion()
        })

    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare_companies_graph', methods=['POST'])
def compare_companies_graph():
    try:
        data = request.json
        name1 = data.get('company1')
        name2 = data.get('company2')

        if not name1 or not name2:
            return jsonify({"error": "Both company names are required."}), 400

        company1 = find_company_match(name1)
        company2 = find_company_match(name2)

        if not company1 or not company2:
            return jsonify({"error": "Couldn't find one or both companies."}), 404

        data1 = {
            "ratings": company1.score,
            "review_count": company1.reviews,
            "keywords": company1.keywords
        }
        data2 = {
            "ratings": company2.score,
            "review_count": company2.reviews,
            "keywords": company2.keywords
        }

        result1 = calculate_eligibility(data1)
        result2 = calculate_eligibility(data2)

        graph_url = generate_comparison_graph(company1, company2, result1, result2)

        better_company = company1.title if result1['probability'] > result2['probability'] else company2.title

        return jsonify({
            "answer": f"{company1.title} is {result1['percentage']} eligible for partnership, while {company2.title} is {result2['percentage']} eligible.",
            "graph_url": graph_url,
            "better_company": better_company,
            "companies": [
                {
                    "name": company1.title,
                    "rating": company1.score,
                    "review_count": company1.reviews,
                    "eligibility_percentage": result1['probability'] * 100
                },
                {
                    "name": company2.title,
                    "rating": company2.score,
                    "review_count": company2.reviews,
                    "eligibility_percentage": result2['probability'] * 100
                }
            ],
            "suggestion": get_new_suggestion()
        })

    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def compare_companies(name1, name2):
    company1 = find_company_match(name1)
    company2 = find_company_match(name2)

    if not company1 or not company2:
        return jsonify({"answer": "Couldn't find one or both companies."})

    data1 = {
        "ratings": company1.score,
        "review_count": company1.reviews,
        "keywords": company1.keywords
    }
    data2 = {
        "ratings": company2.score,
        "review_count": company2.reviews,
        "keywords": company2.keywords
    }

    result1 = calculate_eligibility(data1)
    result2 = calculate_eligibility(data2)

    answer = f"{company1.title} is {result1['percentage']} eligible for partnership, while {company2.title} is {result2['percentage']} eligible."

    return jsonify({
        "answer": answer,
        "suggestion": get_new_suggestion()
    })

def list_eligible_companies():
    eligible_companies = ScrapedCompany.query.all()
    if not eligible_companies:
        return jsonify({
            "answer": "No companies found.",
            "suggestion": get_new_suggestion()
        })

    companies = []
    for company in eligible_companies:
        data = {
            "ratings": company.score,
            "review_count": company.reviews,
            "keywords": company.keywords
        }
        eligibility_result = calculate_eligibility(data)

        # Log the eligibility result
        logger.info(f"Company: {company.title}, Eligibility: {eligibility_result['elegible']}")

        # Update database
        company.elegible = eligibility_result['elegible']
        company.elegibility_precentage = eligibility_result['probability'] * 100  # Store as a percentage
        db.session.add(company)  # Mark for update

        companies.append({
            "name": company.title,
            "eligibility": eligibility_result['percentage'],
            "keywords": [k.strip() for k in company.keywords.split(',')] if company.keywords else [],
            "reviews": company.reviews,
            "rating": company.score,
            "is_eligible": eligibility_result['elegible']
        })

    db.session.commit()  # Commit all changes

    return jsonify({
        "answer": "Here are the companies with their eligibility status:",
        "companies": companies,
        "suggestion": get_new_suggestion()
    })


def reset_predictions():
    for company in ScrapedCompany.query.all():
        company.elegibility_precentage = 0
        company.elegible = False
    db.session.commit()
    return jsonify({"answer": "All predictions have been reset.", "suggestion": get_new_suggestion()})

def show_statistics():
    try:
        total = ScrapedCompany.query.count()
        eligible = ScrapedCompany.query.filter_by(elegible=True).count()
        ineligible = total - eligible

        chart_data = {
            "labels": ["Eligible", "Non-Eligible"],
            "datasets": [{
                "data": [eligible, ineligible],
                "backgroundColor": ["#36A2EB", "#FF6384"],
                "hoverBackgroundColor": ["#36A2EB", "#FF6384"]
            }],
            "type": "pie"  # Specify the chart type here
        }

        return jsonify({
            "answer": f"Total companies: {total}, Eligible companies: {eligible}, Ineligible companies: {ineligible}",
            "chart_data": chart_data,
            "suggestion": get_new_suggestion()
        })
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
