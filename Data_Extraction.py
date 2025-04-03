import requests
import pandas as pd
import time
import os

# Replace with your Adzuna API credentials
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")

# Define base URL template and parameters
RESULTS_PER_PAGE = 100
BASE_URL = "http://api.adzuna.com/v1/api/jobs/in/search/{}"
PARAMS = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "results_per_page": RESULTS_PER_PAGE,
    "content-type": "application/json",
    "max_days_old": 1  # Only fetch jobs from the last 24 hours
}

def fetch_jobs():
    """Fetch job listings from Adzuna API and return as a DataFrame."""
    all_jobs = []
    page = 1

    while True:
        url = BASE_URL.format(page)
        print(f"Fetching page {page}...")

        try:
            response = requests.get(url, params=PARAMS, timeout=10)

            # Retry logic for temporary failures
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                if response.status_code in [500, 502, 503, 504]:
                    print("Retrying after 5 seconds...")
                    time.sleep(5)
                    continue
                break

            data = response.json()
            job_listings = data.get("results", [])

            if not job_listings:
                break

            all_jobs.extend(job_listings)

            # Stop if we have fetched all available pages
            total_count = data.get("count", 0)
            if page * RESULTS_PER_PAGE >= total_count:
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    return pd.DataFrame(all_jobs)

def clean_and_save_data(df):
    """Cleans and saves job data to CSV."""
    if df.empty:
        print("No jobs fetched. Exiting.")
        return

    # Selecting relevant columns
    columns = ['id', 'title', 'redirect_url', 'company', 'location', 'description', 
               'category', 'salary_max', 'contract_type', 'salary_min', 'contract_time']
    df = df.loc[:, columns]

    # Extract "display_name" from nested dictionaries
#     df["location"] = df["location"].apply(lambda x: x.get("display_name", "Unknown") if isinstance(x, dict) else "Unknown")
#     df["category"] = df["category"].apply(lambda x: x.get("label", "Unknown") if isinstance(x, dict) else "Unknown")
    df["location"] = [(ast.literal_eval(company).get("display_name", "Unknown") if isinstance(company, str) else company.get("display_name", "Unknown")) for company in df["location"]]
    df["category"] = [(ast.literal_eval(company).get("label", "Unknown") if isinstance(company, str) else company.get("display_name", "Unknown")) for company in df["category"]]

    # Save raw data
    df.to_csv('raw_data_from_adzuna.csv', mode='a', header=False, index=False)
    
    # Save cleaned data
    df.to_csv('job_listing_data.csv', mode='a', header=False, index=False)
    print(f"✅ {len(df)} job listings appended to CSV.")

# Run the script
df_adzuna = fetch_jobs()
clean_and_save_data(df_adzuna)
