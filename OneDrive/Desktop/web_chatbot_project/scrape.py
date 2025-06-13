import requests
from bs4 import BeautifulSoup

def web_scrap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = soup.get_text(separator= '', strip= True)
    return texts



if __name__ == '__main__' :
    url = "https://inovabeing.com"
    scraped_data = web_scrap(url)
    with open ('data/scraped_data.txt', 'w', encoding= 'utf-8') as f:
        f.write(scraped_data)
    
    print("✅ Website scraped and saved to data/scraped_data.txt")