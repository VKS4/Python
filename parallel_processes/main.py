from bs4 import BeautifulSoup as bsoup
from collections import Counter
import requests
import re
import time
import threading
import matplotlib.pyplot as plt
import nltk

import functions


def download_stopwords():
    # download the stop words for the English language
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    pass


download_stopwords()

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


def language_processing(website_a, website_b, website_c, website_d):
    response_ab = website_a.text + website_b.text
    response_cd = website_c.text + website_d.text
    response_all = response_ab + response_cd

    parsed_content_all = bsoup(response_all, "html.parser")
    text_all = parsed_content_all.get_text()

    # filter out stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words_all = [word for word in re.findall(r'\b\w+\b', text_all) if word.lower() not in stop_words]

    frequency_text_all = Counter(filtered_words_all)

    # get the top 20 words
    frequency_text_20 = frequency_text_all.most_common(20)

    # separate the words and their frequency count
    labels, values = zip(*frequency_text_20)

    # create a bar chart of the top 20 words
    plt.bar(labels, values)

    # set the title and axis labels
    plt.title("20 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Count")

    # rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # record the finish time of the program
    end_time = time.time()

    # calculate the run time of the program
    run_time = end_time - start_time

    # print the run time of the program
    print(f"\nThe program takes {run_time:.2f} seconds to finish.")

    # display the plot
    plt.show()

    pass


language_processing(website_a, website_b, website_c, website_d)
