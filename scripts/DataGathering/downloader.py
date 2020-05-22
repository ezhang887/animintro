import requests

url = 'https://vidstreaming.io/goto.php?url=aHR0cHM6LyAawehyfcghysfdsDGDYdgdsfsdfwstdgdsgtert9zdG9yYWdlLmdvb2dsZWFwaXMuY29tL2ZzZGZlc2ZkL2ZmZi8yM2FfMTU4OTE3MzIxMDEzODkyNC5tcDQ=&title=(HDP%20-%20mp4)%20Kami+no+Tou+Episode+4'

r = requests.get(url, allow_redirects=True)
open('Kami_no_Tou_4.mp4', 'wb').write(r.content)
