# scraping Google Books Ngram (2012 ver.)
# fixed accessai/dynamic_word_embeddings/download_data.py

import logging
import os
import random
import subprocess
import urllib.request

from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_download_links(url):
    resp = urllib.request.urlopen(url)
    # soup = BeautifulSoup(resp, from_encoding=resp.info().get_param("charset"))
    soup = BeautifulSoup(
        resp, from_encoding=resp.info().get_param("charset"), features="html.parser"
    )

    links = []
    for link in soup.find_all("a", href=True):
        url = link["href"]

        if "5gram" in url and "-eng-all-" in url and "-2012" in url:
            links.append(url)

    logging.info(f"Total link counts {len(links)}")

    return links


def accept_links(links, acceptance_rate):
    acceptance_rate /= 100
    accepted_links = []
    for i in range(len(links)):
        if random.random() <= acceptance_rate:
            logging.info(f" [accept_links] accepted!: {links[i].split('/')[-1]}")
            accepted_links.append(links[i])
        else: 
            for target in ["-tv.gz", "-ba.gz", "-co.gz", "-ne.gz", "-nu.gz", "-os.gz", "-pe.gz", "-ph.gz", "-ra.gz", "-te.gz"]:
                if target in links[i]:
                    logging.info(f" [accept_links] accepted!: {links[i].split('/')[-1]}")
                    accepted_links.append(links[i])

    logging.info(" [accept_links] finished!")
    logging.info(f" [accept_links] {len(accepted_links)} links accepted")
    logging.info(f" [accept_links] accepted ratio is {len(accepted_links)/len(links)}")
    return accepted_links


def download_files(links, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for url in tqdm(links, "Downloading..."):
        filename = url.split("/")[-1]
        file_path = os.path.join(output_dir, filename)
        subprocess.run(["curl", url, "--output", file_path])


def main():
    logging.basicConfig(filename="scraping.log", filemode="w", level=logging.INFO)
    url = "http://storage.googleapis.com/books/ngrams/books/datasetsv2.html"
    links = scrape_download_links(url)
    acceptance_rate = int(input("acceptance rate (%): "))
    logging.info(f" [main] acceptance rate: {acceptance_rate} %")
    accepted_links = accept_links(links, acceptance_rate)
    download_files(accepted_links, output_dir="../google-books-ngram")


if __name__ == "__main__":
    main()
