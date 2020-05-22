import time
import json
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options  

"""
Usage
=====

Provide episodes.json - lists gogoanime links with number of episodes

save_all_ids - creates episode_ids.json that lists out all episodes based on
    the initialization parameter links

save_all_download_links - creates download_links.json which converts ids in
    episode_ids.json to actual download links
"""


class LinkScraper():

    def __init__(self, headless=False):
        options = Options()
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(chrome_options=options)
        self.vidstreaming = 'https://vidstreaming.io/download?id='

    """************************************************************
    Basic Functionality
    ************************************************************"""

    def get_video_id(self, link):
        self.driver.get(link)
        time.sleep(1)
        elem = self.driver.find_element_by_class_name('play-video').find_element_by_tag_name('iframe')
        stream_url = elem.get_property('attributes')[0]['nodeValue']
        video_id = re.search(r'id=(.)*&title', stream_url).group()[3:-6]
        return video_id

    def get_download_link(self, ep_id):
        self.driver.get(self.vidstreaming + ep_id)
        while self.driver.title == 'Attention Required! | Cloudflare':
            time.sleep(5)
        elem = self.driver.find_element_by_class_name('dowload').find_element_by_tag_name('a')
        return str(elem.get_attribute('href'))
 
    """************************************************************
    Converting episodes.json -> episode_ids.json
    ************************************************************"""

    def load_episode_list(self, filename='episodes.json'):
        with open(filename, 'r') as f:
            return json.load(f)

    def get_all_ids(self):
        all_ids = self.load_all_ids()
        links = self.load_episode_list()

        for title in links:
            for ep in range(1, links[title]['episodes'] + 1):
                name = title + '_' + str(ep)
                if name in all_ids:
                    print('Already done:', name)
                    continue
                link = links[title]['link'] + str(ep)
                all_ids[name] = str(self.get_video_id(link))
                print('Added:', name)
        return all_ids

    def save_all_ids(self):
        dic = self.get_all_ids()
        with open('episode_ids.json', 'w') as f:
            json.dump(dic, f, indent=2)
            f.close()
    
    def load_all_ids(self, filename='episode_ids.json'):
        with open(filename, 'r') as f:
            return json.load(f)

    """************************************************************
    Converting episode_ids.json -> download_links.json
    ************************************************************"""

    def get_all_download_links(self):
        dic = self.load_all_ids()
        link_dic = {}
        for name in dic:
            download_link = self.get_download_link(dic[name])
            link_dic[name] = download_link
        return link_dic
    
    def save_all_download_links(self):
        dic = self.get_all_download_links()
        with open('download_links.json', 'w') as f:
            json.dump(dic, f, indent=2)

ls = LinkScraper()
ls.save_all_download_links()
# ls.save_all_ids()

