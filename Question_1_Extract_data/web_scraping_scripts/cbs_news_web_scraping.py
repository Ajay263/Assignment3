import csv
import time
import re
import random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cbsnews_politics_scraper.log',
    filemode='w'
)

def clean_text(text):
    """Clean and normalize text."""
    if text:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def initialize_driver():
    """Initialize and configure Chrome WebDriver."""
    chrome_options = Options()
    
    # Comment out headless mode if you want to see the browser for debugging
    chrome_options.add_argument("--headless=new")
    
    # Basic configuration
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--ignore-ssl-errors")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    
    # Fix for SwiftShader WebGL error
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    
    # Use a realistic user agent
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
    
    # Add performance preferences
    chrome_options.add_experimental_option('prefs', {
        'profile.default_content_setting_values.images': 1,
        'disk-cache-size': 4096,
        'profile.default_content_settings.popups': 0,
        'profile.managed_default_content_settings.javascript': 1,
        'credentials_enable_service': False,
        'profile.password_manager_enabled': False
    })
    
    # Setup and return the driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set a page load timeout
    driver.set_page_load_timeout(30)
    
    return driver

def scrape_cbsnews_politics(num_articles=20, max_retries=3):
    """Scrape CBS News Politics articles."""
    articles_data = []
    retry_count = 0
    url = "https://www.cbsnews.com/sanfrancisco/sports/"
    
    while retry_count < max_retries and len(articles_data) == 0:
        try:
            driver = initialize_driver()
            logging.info(f"Navigating to {url}")
            print(f"Navigating to {url}")
            
            # Use get with retry logic
            get_success = False
            for attempt in range(3):
                try:
                    driver.get(url)
                    get_success = True
                    break
                except Exception as e:
                    logging.warning(f"Navigation attempt {attempt+1} failed: {e}")
                    time.sleep(5)
            
            if not get_success:
                raise Exception("Failed to load the page after multiple attempts")
            
            # Wait for content to load with the specific selector
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.XPATH, '//article[contains(@class,"item  item--type-article")]//a'))
                )
                print("Page loaded successfully")
            except TimeoutException:
                logging.error("Timed out waiting for page to load")
                print("Timed out waiting for page content")
                raise
            
            # Scroll down to load more content
            for i in range(10):
                driver.execute_script(f"window.scrollTo(0, {i * 800});")
                time.sleep(0.5)
            
            # Get article links and process them
            articles_data = process_articles(driver, num_articles)
            
        except Exception as e:
            logging.error(f"Attempt {retry_count + 1} failed: {e}")
            print(f"Attempt {retry_count + 1} failed: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
        finally:
            try:
                if 'driver' in locals() and driver:
                    driver.quit()
            except Exception as e:
                logging.error(f"Error closing driver: {e}")
    
    # Save results to CSV
    if articles_data:
        save_to_csv(articles_data)
        logging.info(f"Successfully scraped {len(articles_data)} articles")
        print(f"Successfully scraped {len(articles_data)} articles")
    else:
        logging.error("Failed to scrape any articles after all retries")
        print("Failed to scrape any articles after all retries")
    
    return articles_data

def process_articles(driver, num_articles=20):
    """Process article elements on the CBS News politics page."""
    articles_data = []
    
    # Find all article link elements using the specific selector
    article_links = driver.find_elements(By.XPATH, '//article[contains(@class,"item  item--type-article")]//a')
    logging.info(f"Found {len(article_links)} article links")
    print(f"Found {len(article_links)} article links")
    
    # Get the URLs and headlines before going to each article
    article_urls = []
    for link in article_links[:num_articles]:
        try:
            url = link.get_attribute('href')
            headline = clean_text(link.text)
            if url and url.startswith('https://www.cbsnews.com/'):
                article_urls.append((url, headline))
        except Exception as e:
            logging.warning(f"Error getting link: {e}")
    
    logging.info(f"Collected {len(article_urls)} article URLs to process")
    print(f"Collected {len(article_urls)} article URLs to process")
    
    # Now create a new driver to visit each article
    article_driver = initialize_driver()
    
    try:
        for i, (url, headline) in enumerate(article_urls):
            try:
                logging.info(f"Processing article {i+1}/{len(article_urls)}: {headline}")
                print(f"Processing article {i+1}/{len(article_urls)}: {headline}")
                
                # Get the full article details
                article_data = get_article_details(article_driver, url, headline)
                
                if article_data:
                    articles_data.append(article_data)
                    logging.info(f"Added article: {headline}")
                
                # Add a random delay between articles
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logging.error(f"Error processing article {i+1}: {e}")
                print(f"Error processing article {i+1}: {e}")
    finally:
        try:
            article_driver.quit()
        except:
            pass
    
    return articles_data

def get_article_details(driver, article_url, headline):
    """Get the details from a CBS News article page using the specific selectors."""
    article_data = {
        "headline": headline,
        "article_url": article_url,
        "article_title": "N/A",
        "article_content": "N/A", 
        "published_date": "N/A",
        "writer": "N/A"
    }
    
    try:
        # Load the article page with retry logic
        load_success = False
        for load_attempt in range(3):
            try:
                driver.get(article_url)
                load_success = True
                break
            except Exception as e:
                logging.warning(f"Article load attempt {load_attempt+1} failed: {e}")
                time.sleep(3)
        
        if not load_success:
            logging.error(f"Failed to load article after multiple attempts: {article_url}")
            return article_data
        
        # Wait for the article title to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//h1[contains(@class,"content__title")]'))
            )
            logging.info(f"Article page loaded: {article_url}")
        except TimeoutException:
            logging.warning(f"Timeout waiting for article title: {article_url}")
            # Continue anyway to try to get content
        
        # Get article title
        try:
            title_element = driver.find_element(By.XPATH, '//h1[contains(@class,"content__title")]')
            article_data["article_title"] = clean_text(title_element.text)
        except NoSuchElementException:
            logging.warning(f"Could not find title element for article: {article_url}")
        
        # Try to get published date
        try:
            date_element = driver.find_element(By.XPATH, '//p[contains(@class,"content__meta content__meta--timestamp")]')
            article_data["published_date"] = clean_text(date_element.text)
        except NoSuchElementException:
            logging.warning(f"Could not find date element for article: {article_url}")
        
        # Try to get writer/author
        try:
            writer_element = driver.find_element(By.XPATH, '//span[contains(@class,"byline__author__popover-btn__label underline-on-hover")]')
            article_data["writer"] = clean_text(writer_element.text)
        except NoSuchElementException:
            logging.warning(f"Could not find writer element for article: {article_url}")
        
        # Get article content paragraphs
        try:
            content_elements = driver.find_elements(By.XPATH, '//section[contains(@class,"content__body")]//p')
            
            if content_elements:
                article_content = " ".join([clean_text(p.text) for p in content_elements if p.text.strip()])
                article_data["article_content"] = article_content
            else:
                logging.warning(f"No content elements found for article: {article_url}")
                
        except Exception as e:
            logging.error(f"Error getting content for article: {e}")
        
    except Exception as e:
        logging.error(f"Error processing article {article_url}: {e}")
    
    return article_data

def save_to_csv(articles_data):
    """Save scraped data to CSV file."""
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'cbsnews_politics_{timestamp}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["headline", "article_url", "article_title", "article_content", "published_date", "writer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in articles_data:
                writer.writerow(article)
        logging.info(f"Data successfully saved to {filename}")
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    try:
        print("Starting scraper to collect CBS News politics articles...")
        num_articles_to_scrape = 25
        articles = scrape_cbsnews_politics(num_articles=num_articles_to_scrape)
        
        if articles:
            print(f"Completed! Scraped {len(articles)} articles with full content.")
        else:
            print("Failed to scrape articles from CBS News politics section.")
        
    except Exception as e:
        logging.error(f"Main execution error: {e}")
        print(f"Critical error: {e}")