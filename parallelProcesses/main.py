import requests
from bs4 import BeautifulSoup as bsoup
import re
import time


# start timing the program
start_time = time.time()

# scap the raw HTML data from two websites
website_a = requests.get("https://www.idnes.cz/zpravy")
website_b = requests.get("https://www.seznamzpravy.cz/")

# save the raw HTML data to response_a and response_b variables
response_a = website_a.text
response_b = website_b.text

parsed_content_a = bsoup(response_a, "html.parser")
parsed_content_b = bsoup(response_b, "html.parser")

text_a = bsoup.getText(parsed_content_a)
text_b = bsoup.getText(parsed_content_b)

# obtain words from scrapped data by using BeautifulSoup and re libraries
text_a_words = re.findall(r'\b\w+\b', text_a)
text_b_words = re.findall(r'\b\w+\b', text_b)

print("printing the raw data from website_a \n {}" .format(text_a_words))
print("printing the raw data from website_b \n {}" .format(text_b_words))

end_time = time.time()
run_time = end_time - start_time

print(f"The program takes {run_time:.2f} seconds to finish.")

# numpy.savetxt("website_a.csv")
# numpy.savetxt("website_b.csv")