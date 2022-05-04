import requests
import sys
import json
import os
url=""
# for k, v in sorted(os.environ.items()):
#     print(k+':', v)
# print('\n')
try:
    k = os.environ['WEBHOOK']
    url = "https://discord.com/api/webhooks/{hook}".format(hook=k)
except:
    print("on system or cant findsecret")
    f = open("/home/whorehay/Desktop/github /githubintragrationdiscordbot/creds.json",)
    data = json.load(f)
    f.close()
    url = "https://discord.com/api/webhooks/{hook}".format(hook=data["webhook"])
n = len(sys.argv)
k = "None"
if(n>1):
     k = sys.argv[1]

if str(k) =="0":
    k+="::(✔️)::"
else:
    k+="::(X)::"
Message = {
    "content": "Done Testing. \nstatus: {code}".format(code=k)
}
requests.post(url, data=Message)
