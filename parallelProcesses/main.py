import requests
import bs4
import re


website_a = requests.get("https://www.idnes.cz/zpravy")
website_b = requests.get("https://www.seznamzpravy.cz/")

response_a = website_a.text
response_b = website_b.text

print("printing the raw data from website_a \n {}" .format(response_a))
print("printing the raw data from website_b \n {}" .format(response_b))

# numpy.savetxt("website_a.csv")
# numpy.savetxt("website_b.csv")