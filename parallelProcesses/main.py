from bs4 import BeautifulSoup as bsoup
from collections import Counter
import requests
import re
import time
import threading
import matplotlib

import functions

# start timing the program
start_time = time.time()


def thread_a_result():
    thread_a.result = functions.scrap_data_a()


def thread_b_result():
    thread_b.result = functions.scrap_data_b()


def thread_c_result():
    thread_c.result = functions.scrap_data_c()


def thread_d_result():
    thread_d.result = functions.scrap_data_d()


thread_a = threading.Thread(target=thread_a_result)
thread_b = threading.Thread(target=thread_b_result)
thread_c = threading.Thread(target=thread_c_result)
thread_d = threading.Thread(target=thread_d_result)

thread_a.start()
thread_b.start()
thread_c.start()
thread_d.start()

thread_a.join()
thread_b.join()
thread_c.join()
thread_d.join()

website_a = thread_a.result
website_b = thread_b.result
website_c = thread_c.result
website_d = thread_d.result

# save the raw HTML data to response_a and response_b variables
# website_a = functions.scrap_data_a()
# website_a = functions.scrap_data_b()
# website_a = functions.scrap_data_c()
# website_a = functions.scrap_data_d()

response_ab = website_a.text + website_b.text
response_cd = website_c.text + website_d.text
response_all = response_ab + response_cd

# website_a.text + website_b.text + website_c.text + website_d.text
# print("Prnting stuff {}" .format(response_a))

parsed_content_all = bsoup(response_all, "html.parser")
print("Prnting stuff {}".format(response_all))

text_all = bsoup.getText(parsed_content_all)

# filter words from HTML
text_words_all = re.findall(r'\b\w+\b', text_all)

# filter out numbers
filtered_words_all = [word for word in text_words_all if not re.search("\d", word)]

# print the filtered words
print("\nPrinting filtered words\n {}".format(filtered_words_all))

frequency_text_all = Counter(filtered_words_all)

print("\nPrinting the frequency of the words found in filtered_words_all variable\n {}".format(frequency_text_all))
# #
# # # print("\nprinting the word frequency \n {}" .format(frequency_text_ab))
# #
# # # record the finish time of the program
# # end_time = time.time()
# #
# # # calculate the run time of the program
# # run_time = end_time - start_time
# #
# # # print the run time of the program
# # print(f"\nThe program takes {run_time:.2f} seconds to finish.")
# #
# # # numpy.savetxt("website_a.csv")
# # # numpy.savetxt("website_b.csv")
