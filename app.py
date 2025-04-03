import os
import pickle
import time
import json
import csv
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import stanza
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch
import threading
from supabase import create_client, Client

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Replace with your Supabase URL
SUPABASE_KEY = os.getenv("SUPABASE_KEY")                   # Replace with your Supabase API key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


SESSION_FILE = "session_details.json"

def load_session():
    if not os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "w") as f:
            json.dump({}, f)
    with open(SESSION_FILE, "r") as f:
        return json.load(f)

def save_session(data):
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=4)


# --- Logging & Flask App Setup ---
logging.basicConfig(filename="chatbot.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session management
CORS(app)

# Ensure the app runs on the correct host and port
#port = int(os.environ.get("PORT", 8080))
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# --- NLP and Semantic Model Initialization ---
nlp = stanza.Pipeline('en', processors='tokenize,pos,ner', verbose=False)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory stores for chatbot conversation histories and ambiguous selections
session_store = {}
ambiguous_store = {}

# Global flag for semantic embeddings readiness.
embeddings_ready = False

# -------------------- Data Loading & Semantic Embedding --------------------

def load_job_listings_from_csv():
    listings = []
    try:
        with open("job_listing_data.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                listings.append(row)
        logging.info("Loaded %d job listings from CSV.", len(listings))
    except Exception as e:
        logging.error("Error reading CSV: %s", e)
    return listings

def load_session_details():
    try:
        with open("session_details.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info("Loaded session details from JSON.")
        return data
    except Exception as e:
        logging.error("Error reading JSON: %s", e)
    return {}

def fetch_job_listings_scrape():
    url = "https://test1.jobsforher.com/jobs"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            listings = []
            job_cards = soup.find_all("div", class_="job-card")
            for card in job_cards:
                title = card.find("h2").get_text(strip=True) if card.find("h2") else "No Title"
                company = card.find("span", class_="company").get_text(strip=True) if card.find("span", class_="company") else "Unknown Company"
                listings.append({"title": title, "company": company})
            logging.info("Scraped %d job listings from %s", len(listings), url)
            if listings:
                return listings
        else:
            logging.warning("Scraping failed with status code: %s", response.status_code)
    except Exception as e:
        logging.error("Exception during scraping: %s", e)
    return load_job_listings_from_csv()

def fetch_job_listings_api():
    try:
        response = requests.get("https://api.example.com/jobs", timeout=5)
        if response.status_code == 200:
            logging.info("Fetched job listings from external API.")
            return response.json()
        else:
            logging.warning("API call unsuccessful. Status code: %s", response.status_code)
    except Exception as e:
        logging.error("Exception during API call: %s", e)
    return load_job_listings_from_csv()

job_listings_static = load_job_listings_from_csv()
session_details = load_session_details()

# def build_job_embeddings():
#     global embeddings_ready
#     cache_file = "job_embeddings.pkl"
#     if os.path.exists(cache_file):
#         try:
#             with open(cache_file, "rb") as f:
#                 cached_embeddings = pickle.load(f)
#             if len(cached_embeddings) == len(job_listings_static):
#                 for i, job in enumerate(job_listings_static):
#                     job['embedding'] = cached_embeddings[i]
#                 logging.info("Loaded cached job embeddings.")
#                 embeddings_ready = True
#                 return
#             else:
#                 logging.info("Cache length mismatch. Recomputing embeddings.")
#         except Exception as e:
#             logging.error("Error loading cached embeddings: %s", e)
#     embeddings = []
#     for job in job_listings_static:
#         text = ""
#         for field in ["title", "company", "description", "category"]:
#             if field in job and job[field]:
#                 text += job[field] + " "
#         embedding = semantic_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
#         job['embedding'] = embedding
#         embeddings.append(embedding)
#     try:
#         with open(cache_file, "wb") as f:
#             pickle.dump(embeddings, f)
#         logging.info("Computed and cached job embeddings.")
#     except Exception as e:
#         logging.error("Error caching embeddings: %s", e)
#     embeddings_ready = True
def build_job_embeddings():
    global embeddings_ready
    cache_file = "job_embeddings.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_embeddings = pickle.load(f)
            if len(cached_embeddings) == len(job_listings_static):
                for i, job in enumerate(job_listings_static):
                    job['embedding'] = cached_embeddings[i]
                logging.info("Loaded cached job embeddings.")
                embeddings_ready = True
                return
            else:
                logging.info("Cache length mismatch. Recomputing embeddings.")
        except Exception as e:
            logging.error("Error loading cached embeddings: %s", e)
    embeddings = []
    for job in job_listings_static:
        text = ""
        for field in ["title", "company", "description", "category"]:
            if field in job and job[field]:
                text += job[field] + " "
        embedding = semantic_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        job['embedding'] = embedding
        embeddings.append(embedding)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info("Computed and cached job embeddings.")
    except Exception as e:
        logging.error("Error caching embeddings: %s", e)
    embeddings_ready = True
    
# Run embedding computation in a background thread.
threading.Thread(target=build_job_embeddings, daemon=True).start()

def wait_for_embeddings(timeout=10):
    start_time = time.time()
    while not embeddings_ready and time.time() - start_time < timeout:
        time.sleep(0.1)
    if not embeddings_ready:
        logging.warning("Embeddings not ready after waiting for %s seconds.", timeout)

# -------------------- Chatbot Core Functionality --------------------

def detect_bias(message):
    biased_terms = ["only man", "not for women", "typical male", "stereotype"]
    for term in biased_terms:
        if term in message.lower():
            return True
    return False

def search_jobs(query, similarity_threshold=0.3):
    wait_for_embeddings()
    query_embedding = semantic_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    scored_jobs = []
    for job in job_listings_static:
        score = util.cos_sim(query_embedding, job['embedding'])
        scored_jobs.append((score.item(), job))
    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    matches = [job for score, job in scored_jobs if score >= similarity_threshold]
    logging.info("Semantic search found %d matching jobs for query: %s", len(matches), query)
    return matches

def get_detail_from_job(job, detail_type):
    if detail_type == "link":
        return f"Here is the job link: {job.get('redirect_url', 'No link available')}."
    elif detail_type == "salary":
        salary_min = job.get("salary_min", "N/A")
        salary_max = job.get("salary_max", "N/A")
        contract_type = job.get("contract_type", "N/A")
        return f"The salary range is {salary_min} - {salary_max} ({contract_type})."
    elif detail_type == "skills":
        return f"Job description/skills: {job.get('description', 'No description available')}."
    elif detail_type == "experience":
        return "Experience details are not available for this job."
    elif detail_type == "contract time":
        return f"The contract time is {job.get('contract_time', 'N/A')}."
    else:
        return "I'm sorry, I didn't understand what detail you need."

def get_job_detail(query, detail_type, session_id):
    matches = search_jobs(query)
    if not matches:
        return "Sorry, I couldn't find that job."
    if len(matches) == 1:
        job = matches[0]
        return get_detail_from_job(job, detail_type)
    else:
        ambiguous_store[session_id] = {"matches": matches, "detail_type": detail_type}
        response = "I found multiple jobs that match. Please specify by entering the number:\n"
        for i, candidate in enumerate(matches[:3]):
            response += f"{i+1}. {candidate.get('title', 'No Title')} at {candidate.get('company', 'Unknown Company')}\n"
        return response

def process_message(message, history, session_id):
    if message.strip().isdigit() and session_id in ambiguous_store:
        index = int(message.strip()) - 1
        data = ambiguous_store.pop(session_id)
        matches = data["matches"]
        detail_type = data["detail_type"]
        if index < 0 or index >= len(matches):
            return "Invalid selection. Please try again."
        selected_job = matches[index]
        return get_detail_from_job(selected_job, detail_type)
    
    doc = nlp(message)
    tokens = []
    entities = []
    for sentence in doc.sentences:
        tokens.extend([word.text for word in sentence.words])
        entities.extend([(ent.text, ent.type) for ent in sentence.ents])
    logging.info("Processed message. Tokens: %s | Entities: %s", tokens, entities)
    
    if detect_bias(message):
        return "I detected a potentially biased query. Let’s keep our conversation positive and inclusive."
    
    message_lower = message.lower()
    if any(keyword in message_lower for keyword in ["link", "salary", "skill", "experience", "contract time"]):
        if "link" in message_lower:
            return get_job_detail(message, "link", session_id)
        elif "salary" in message_lower:
            return get_job_detail(message, "salary", session_id)
        elif "skill" in message_lower:
            return get_job_detail(message, "skills", session_id)
        elif "experience" in message_lower:
            return get_job_detail(message, "experience", session_id)
        elif "contract time" in message_lower:
            return get_job_detail(message, "contract time", session_id)
    
    if "job" in message_lower or "career" in message_lower:
        matches = search_jobs(message)
        if matches:
            response = "Here are some job listings that match your query:\n"
            for job in matches[:3]:
                response += f"- {job.get('title', 'No Title')} at {job.get('company', 'Unknown Company')}\n"
            return response
        else:
            return "Sorry, no job listings match your query right now."
    elif "session" in message_lower or "event" in message_lower:
        if session_details:
            response = "Upcoming sessions:\n"
            for key, detail in session_details.items():
                response += f"- {key}: {detail}\n"
            return response
        else:
            return "Sorry, session details are not available."
    elif "help" in message_lower:
        return ("I can help with job listings, session details, mentorship opportunities, "
                "and career advice. What would you like to know?")
    elif "faq" in message_lower:
        return ("FAQs:\n- How do I apply for a job?\n- How do I register for a session?\n"
                "- Who can join the mentorship program?")
    else:
        return "I'm sorry, I didn't understand that. Could you please clarify?"

# -------------------- Authentication with Supabase --------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')

        try:
            # Sign up user
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            logging.info("Supabase signup response: %s", response)

            # Check if there's an error
            if response.user is None:
                return render_template('signup.html', error="Signup failed. Please check your details.")

            # Store user session
            session["user"] = response.user.id  # Store user ID in session

            # Redirect to main chatbot UI after signup
            return redirect(url_for('index'))
        except Exception as e:
            logging.error("Error during signup: %s", e)
            return render_template('signup.html', error="Signup failed. Please try again.")

    return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')

#         try:
#             # Sign in user
#             response = supabase.auth.sign_in_with_password({
#                 "email": email,
#                 "password": password
#             })

#             logging.info("Supabase login response: %s", response)

#             # Check if login was successful
#             if response.user is None:
#                 return render_template('login.html', error="Invalid email or password.")

#             # Store user session
#             session["user"] = response.user.id  # Store user ID in session

#             # Redirect to main chatbot UI after login
#             return redirect(url_for('index'))
#         except Exception as e:
#             logging.error("Error during login: %s", e)
#             return render_template('login.html', error="Login failed. Please try again.")

#     return render_template('login.html')
#  Fix: Allow GET & POST for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')  # Ensure 'login.html' exists

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return "Missing email or password", 400

    # Login request to Supabase
    response = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        json={"email": email, "password": password},
        headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
    )

    data = response.json()
    if response.status_code == 400 and "Email not confirmed" in str(data):
        return "Email not confirmed. Please check your inbox.", 403
    elif response.status_code != 200:
        return "Invalid login credentials", 401

    session['user'] = {"email": email, "access_token": data.get("access_token")}
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/welcome')
def welcome():
    user = session.get("user")
    return render_template('welcome.html', user=user)

@app.route('/faq')
def faq():
    return render_template('faq.html')

# @app.route("/profile")
# def profile():
#     user_email = session.get("email")
#     if not user_email:
#         return redirect(url_for("login"))
    
#     session_data = load_session()
#     past_searches = session_data.get(user_email, [])
#     return jsonify({"email": user_email, "past_searches": past_searches})
# Fix: Profile Page
@app.route('/profile', methods=['GET'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']['email']
    
    try:
        with open("session_details.json", "r") as file:
            session_data = json.load(file)
    except FileNotFoundError:
        session_data = {"past_searches": []}

    return jsonify({
        "email": user_email,
        "past_searches": session_data.get("past_searches", [])
    })

# Fix: Create 'session_details.json' if missing
def ensure_session_file():
    try:
        with open("session_details.json", "r") as file:
            json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open("session_details.json", "w") as file:
            json.dump({"past_searches": []}, file)
# -------------------- Main Chatbot Routes --------------------

@app.route('/')
# def index():
#     return redirect(url_for('welcome'))
def index():
    user = session.get("user")
    if user:
        # Render chatbot UI if logged in
        return render_template('index.html', user=user)
    else:
        # Otherwise, redirect to login
        return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', ' ')
        session_id = data.get('session_id', 'default')
        logging.info("Received message from session %s: %s", session_id, user_message)
        
        history = session_store.get(session_id, [])
        history.append({"sender": "user", "message": user_message, "timestamp": datetime.now().isoformat()})
        
        bot_response = process_message(user_message, history, session_id)
        history.append({"sender": "bot", "message": bot_response, "timestamp": datetime.now().isoformat()})
        session_store[session_id] = history
        
        logging.info("Updated session %s history. Total messages: %d", session_id, len(history))
        return jsonify({"response": bot_response})
    except Exception as e:
        logging.error("Error in /chat endpoint: %s", e)
        return jsonify({"response": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    ensure_session_file()    
    app.run(debug=True)

# to run this locally run - python app.py and access the app on "http://127.0.0.1:5000/"
 