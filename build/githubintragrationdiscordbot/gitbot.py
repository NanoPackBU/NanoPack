import discord
import json
from github import Github

f = open("/home/whorehay/Desktop/github /githubintragrationdiscordbot/creds.json",)
data = json.load(f)
f.close()
client = discord.Client()

def send():
    github_token = data["devTokenGit"]
    g = Github(github_token)
    g.get_user("BostonUniversitySeniorDesign").get_repo("NanoView_G33").create_repository_dispatch("start-workflow")

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!build'):
        await message.channel.send('Sending Request')
        send()
        await message.channel.send('Attempt Made')
    if message.content.startswith('!selfdestruct'):
        await message.channel.send('I can\'t let you do that {name}'.format(name=message.author))   


client.run(data["devTokenDiscord"])
