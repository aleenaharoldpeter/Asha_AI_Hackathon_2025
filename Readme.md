# Asha AI Chatbot

## Overview

Asha AI Chatbot is an intelligent, context-aware virtual assistant designed to enhance user engagement on the JobsForHer Foundation platform. It helps users discover job listings, community events, mentorship programs, and provides detailed information (such as salary, required skills, and job links) in a conversational manner. Additionally, the chatbot supports user registration, login, and profile management, making it a comprehensive resource for professional growth and networking—especially aimed at empowering women in their professional journey.

## Problem Statement

In today’s fast-evolving digital world, seamless and intelligent conversations are key to enhancing user engagement. The Asha AI Chatbot is built to transform how users interact with the JobsForHer Foundation platform by:
- Guiding users to explore job listings, events, and mentorship programs.
- Assisting in user signups and profile updates.
- Addressing frequently asked questions (FAQs).
- Delivering accurate, real-time responses through retrieval-augmented generation (RAG) and semantic search.
- Ensuring ethical AI practices by mitigating gender bias and promoting inclusivity.

## Demo Video
- [Click here to view the demo video](https://www.youtube.com/watch?v=NXH02aUOXEw&feature=youtu.be)

## Features Covered

- **Contextual Awareness & Multi-Turn Conversations:**  
  Maintains conversation history and handles follow-up queries related to previously mentioned jobs.
  
- **Semantic Search & RAG:**  
  Uses SentenceTransformers to compute embeddings and perform semantic search for highly relevant job listings.

- **Detailed Job Information:**  
  Provides job details such as links, salary ranges, skills/description, and contract time. Handles ambiguous queries by prompting the user to select from multiple matches.

- **User Registration, Login & Profile Management:**  
  Dummy pages for signup, login, profile, and FAQs are provided to simulate a full user lifecycle.

- **Real-Time Data Retrieval:**  
  Integrates data from CSV files and real-time web scraping as a fallback mechanism.

- **Ethical AI & Bias Prevention:**  
  Includes a bias detection mechanism to ensure inclusive and responsible responses.

- **Robust Error Handling & Logging:**  
  Comprehensive logging and error handling for smooth operation and easier troubleshooting.

## Prerequisites

- **Python 3.13** (or a compatible version)
- **pip** (Python package installer)
- A Supabase account for authentication (set your Supabase URL and API key as environment variables on your hosting platform)

## Technology Stack

- **Backend Framework:** Flask  
- **NLP:** Stanza  
- **Semantic Search:** SentenceTransformers (using the `all-MiniLM-L6-v2` model)  
- **Database & Authentication:** Supabase (for user signup, login, and profile management)  
- **Frontend:** HTML, Bootstrap (for responsive UI design)  
- **Data Sources:** CSV files, JSON files, and real-time web scraping  
- **Deployment:** Free hosting platforms such as Render, Railway, or Hugging Face Spaces

## Installation and Running Locally

1. **Clone the Repository:**
   ```sh
   https://github.com/aleenaharoldpeter/Asha_AI_Hackathon_2025.git
   ```
2. **Create and Activate a Virtual Environment:**
    ```sh
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3. **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
4. **Set Up Environment Variables (for Supabase):** Create a .env file or set them directly in your hosting platform:
    ```sh
    export SUPABASE_URL="https://your-supabase-url.supabase.co"
    export SUPABASE_KEY="your_supabase_api_key"
    ```
5. **Run the Application:**
    ```sh
    python app.py
    ```
    The app will run locally at http://127.0.0.1:5000/.

## Project Structure
```bash
├── app.py
├── requirements.txt
├── session_details.json         # Contains event/mentorship data and past searches
├── job_listing_data.csv         # CSV file with job listings data
├── job_embeddings.pkl           # Cached embeddings for job listings (generated automatically)
└── templates/                   # HTML templates folder
    ├── index.html               # Main chatbot interface with navigation
    ├── signup.html              # User signup page
    ├── login.html               # User login page
    ├── welcome.html             # Welcome page after signup
    ├── profile.html             # Dummy profile page
    └── faq.html                 # Frequently Asked Questions page
```
## Usage    
- **Chatbot Interface:**

    After logging in, users are redirected to the main chat interface where they can type queries like "Data Engineer Jobs" or "salary for [Job Title]". The chatbot uses semantic search and context handling to provide relevant responses and detailed job information.
- **User Signup and Login:**

    Navigate to `/signup` to register and `/login` to sign in. These pages simulate user authentication via Supabase.
- **Profile and FAQ Pages:**

    Access `/profile` to view a dummy profile and `/faq` to see common questions and answers.
## Deployment
- **Push your code to GitHub.**
- **Deploy on a Free Platform:**

    - **Render:** Create a new web service, connect your GitHub repo, and set the start command to `gunicorn app:app`.    
    - **Railway or Hugging Face Spaces:** Follow their respective instructions for Python/Flask deployments.

- **Configure Environment Variables:**

    Set your `SUPABASE_URL` and `SUPABASE_KEY` on your hosting platform to keep credentials secure.
## Additional Information
- The chatbot leverages retrieval-augmented generation (RAG) and semantic search for accurate, real-time responses.
- All components are built using free, open-source tools.
- Detailed logging and error handling have been implemented to support future analytics and continuous improvement.
    
