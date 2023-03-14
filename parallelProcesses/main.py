from bs4 import BeautifulSoup as bsoup
# from collections import Counter
import requests
import re
import time
# import matplotlib


# start timing the program
start_time = time.time()

# scap the raw HTML data from two websites
website_a = requests.get("https://www.idnes.cz/zpravy")
website_b = requests.get("https://www.seznamzpravy.cz/")
website_c = requests.get("https://www.novinky.cz/")
website_d = requests.get("https://zpravy.aktualne.cz/")

# save the raw HTML data to response_a and response_b variables
response_a = website_a.text
response_b = website_b.text
response_c = website_c.text
response_d = website_d.text

parsed_content_a = bsoup(response_a, "html.parser")
parsed_content_b = bsoup(response_b, "html.parser")
parsed_content_c = bsoup(response_c, "html.parser")
parsed_content_d = bsoup(response_d, "html.parser")

text_a = bsoup.getText(parsed_content_a)
text_b = bsoup.getText(parsed_content_b)
text_c = bsoup.getText(parsed_content_c)
text_d = bsoup.getText(parsed_content_d)

text_total = []

text_total.append(text_a)
text_total.append(text_b)
text_total.append(text_c)
text_total.append(text_d)



print("\nprinting resulting list text_total made by appending lists text_a, b, c, and d \n {}" .format(text_total))

# obtain words from scrapped data by using BeautifulSoup and re libraries
# text_words = re.findall(r'\b\w+\b', text_total)

# pattern = r'\b\w+\b'
# text_words = []
#
# for string in text_total:
#     result = re.findall(pattern, text_total)
#     text_words.append(result)
#
# print("printing the words from website_a,b,c, and d \n {}" .format(text_words))
#
# filtered_words = [word for word in text_words if not re.search("\d", word)]
#
# print("\nprinting further filtered list of words scrapped from the websites excluding numbers \n {}"
#       .format(filtered_words))
#
# # frequency_text_ab = Counter()
#
# # print("\nprinting the word frequency \n {}" .format(frequency_text_ab))
#
# # record the finish time of the program
# end_time = time.time()
#
# # calculate the run time of the program
# run_time = end_time - start_time
#
# # print the run time of the program
# print(f"\nThe program takes {run_time:.2f} seconds to finish.")
#
# # numpy.savetxt("website_a.csv")
# # numpy.savetxt("website_b.csv")