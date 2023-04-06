# import re
# import main
import requests


def scrap_data_a():
    website_a = requests.get("https://www.nytimes.com/international/")

    return website_a


def scrap_data_b():
    website_b = requests.get("https://edition.cnn.com/")

    return website_b


def scrap_data_c():
    website_c = requests.get("https://www.bbc.com/news//world")

    return website_c


def scrap_data_d():
    website_d = requests.get("https://time.com/")

    return website_d
