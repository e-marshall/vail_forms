from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

class SECScraper:
    def __init__(self, target_url):
        self.target_url = target_url
        self.headers = {"User-Agent": "Your Name/Your App (your_email@example.com)"}
        self.base_url = "https://www.sec.gov/Archives/edgar/data/812011/"
        self.soup_dict = {}

    def fetch_data(self):
        response = requests.get(self.target_url, headers=self.headers)
        data = response.json()
        recent_filings_dict = data["filings"]["recent"]
        recent_filings_df = pd.DataFrame(recent_filings_dict)
        recent_filings_df['filingDate_dt'] = pd.to_datetime(recent_filings_df['filingDate'])
        self.recent_filings = recent_filings_df.loc[recent_filings_df['filingDate_dt'] > self.cutoff_date]

    def scrape_filings(self):
        for accession, document, filing_date in zip(self.recent_filings["accessionNumber"], self.recent_filings["primaryDocument"], self.recent_filings['filingDate']):
            accession_folder = accession.replace("-", "")
            filing_url = f"{self.base_url}{accession_folder}/{document}"
            print('filing url: ',filing_url)
            print('accession: ', accession)
            print('filing date: ', filing_date)
            filing_response = requests.get(filing_url, headers=self.headers)
            soup = BeautifulSoup(filing_response.content, 'lxml')
            self.soup_dict[filing_date] = soup

    def run(self, cutoff_date, form_type):
        self.cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
        self.fetch_data()
        self.recent_filings = self.recent_filings.loc[self.recent_filings['form'] == form_type]
        self.scrape_filings()
        return self.soup_dict
