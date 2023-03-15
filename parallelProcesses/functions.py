# import re
# import main
import requests


def scrap_data_a():
    website_a = requests.get("https://www.idnes.cz/zpravy")

    return website_a


def scrap_data_b():
    website_b = requests.get("https://www.seznamzpravy.cz/")

    return website_b


def scrap_data_c():
    website_c = requests.get("https://www.novinky.cz/")

    return website_c


def scrap_data_d():
    website_d = requests.get("https://zpravy.aktualne.cz/")

    return website_d
