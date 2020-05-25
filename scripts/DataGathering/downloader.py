import os
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
        print(filename, 'already downloaded')
        continue
    
    try:
        url = download_links[episode]
        r = requests.get(url, allow_redirects=True)
        open(folder + filename, 'wb').write(r.content)
        
        print('Finished:', episode)

    except:
        print('Failed:', episode)
