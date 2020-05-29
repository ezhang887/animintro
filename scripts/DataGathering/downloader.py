import os
import sys
import json
import requests

folder = 'Episodes/'
downloaded = os.listdir(folder)

download_links = {}
with open('download_links.json', 'r') as f:
    download_links = json.load(f)
    f.close()

for episode in download_links:
    filename = episode + '.mp4'
    if filename in downloaded:
        # print(filename, 'already downloaded')
        continue
    
    try:
        url = download_links[episode]
        r = requests.get(url, allow_redirects=True, stream=True)

        # progress bar
        total_length = int(r.headers.get('content-length'))
        dl = 0
        print("Downloading", episode)
        
        with open(folder + filename, 'wb') as f:
            for data in r.iter_content(chunk_size=4096):
                f.write(data)
                dl += len(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


        print('\nFinished:', episode)

    except:
        print('Failed:', episode)
