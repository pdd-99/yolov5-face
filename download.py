import requests
URL = "https://instagram.com/favicon.ico"
response = requests.get(URL)
open("instagram.ico", "wb").write(response.content)