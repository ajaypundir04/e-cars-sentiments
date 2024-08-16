from selenium import webdriver
import os

def survey_init():
    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome()

    # Get the absolute path of the HTML file
    file_path = os.path.abspath("survey.html")
    
    # Load the local HTML file
    driver.get(f"file://{file_path}")
    
    # Print the page source
    print(driver.page_source.encode('utf-8'))
    
    # Close the browser
    driver.quit()
