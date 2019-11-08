#!/usr/bin/env python
# coding: utf-8

# In[370]:


import re
import urllib.request
from bs4 import BeautifulSoup
import numpy as np


# In[342]:


import pandas as pd
from sqlalchemy import create_engine
import time
import random
import matplotlib.pyplot as plt


# In[454]:


player_values_url = "https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop"
url2 = "?page="

japan_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=77&yt0=Show"
china_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=34&yt0=Show"
korea_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=87&yt0=Show"

ua_headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'
}


# In[470]:


# import http.client
# http.client.HTTPConnection._http_vsn = 10
# http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

def getHtml(url, headers):
    
    time.sleep(5)
    
    print(url)
    request = urllib.request.Request(url, headers = headers)
    response = urllib.request.urlopen(request, timeout = 50)
    print(response)
    
    try:
        html = response.read()
        html = html.decode("utf-8")
    except (http.client.IncompleteRead) as e:
        html = e.partial
    return html


# In[457]:


# request = urllib.request.Request(player_values_url, headers=ua_headers)
# response = urllib.request.urlopen(request, timeout=10)
# html = response.read()
# html = html.decode("utf-8")
# html


# In[325]:


def getData(html):
    data = []
    soup=BeautifulSoup(html)
    div = soup.find('div', attrs={'class':'responsive-table'})
    table_body = div.find('tbody')

    rows = table_body.find_all('tr')[::3]
    
    print("parsing data......")
    
    for row in rows:
        players = row.find_all('td')
        player = [x.text.strip() for x in players]

        alt = row.find_all('img')
        player.extend([a.attrs['alt'] for a in alt][1:3])
        
        data.append(player)
    
    return data


# In[249]:


# h1 = soup.find('thead')
# h2 = h1.find_all('th')
# col_name = [head.text.strip() for head in h2]
# col_name


# In[329]:


def to_df(data):
    df = pd.DataFrame(data)
    df_dropped = df.drop([1,2,6,7],axis="columns")
    df_dropped.columns = ['#', 'Player','Position', 'Age', 'Market value','Nat.', 'Club']
    df_dropped.set_index('#')
    
    return df_dropped


# In[468]:


def to_sql(df, db_name):
    df.set_index('#')
    engine = create_engine('sqlite:///' + db_name + '.sqlite')
    df.to_sql(db_name, con = engine, if_exists='append')
    print("saved......")


# In[288]:


df = pd.DataFrame(data)
df_dropped = df.drop([1,2,6,7],axis="columns")
df_dropped.columns = ['#', 'Player','Position', 'Age', 'Market value','Nat.', 'Club']
df_dropped.set_index('#')


# In[331]:


for i in range(1,11):
    
    html = getHtml(player_values_url + url2 + str(i), ua_headers)
    
    data = getData(html)
    
    df = to_df(data)
    
    print(df)
    
    to_sql(df, 'PlayerValue')

print("finish!!!")


# In[475]:


for url in [japan_player_values_url, china_player_values_url, korea_player_values_url]:
    
    html = getHtml(player_values_url + url, ua_headers)
    
    data = getData(html)
    
    df = to_df(data)
    
    print(df)
    
    to_sql(df, 'EastAsianPlayerValue')

print("finish!!!")


# In[336]:


engine = create_engine('sqlite:///PlayerValue.sqlite')

# engine.execute("SELECT * FROM PlayerValue").fetchall()
df_all = pd.read_sql_query("SELECT * FROM PlayerValue", engine)


# In[489]:


engine1 = create_engine('sqlite:///EastAsianPlayerValue.sqlite')

# engine.execute("SELECT * FROM PlayerValue").fetchall()
df_asian = pd.read_sql_query("SELECT * FROM EastAsianPlayerValue", engine1)


# In[516]:


def convert_int(str_0):
    if ',' in str_0:
        str_1 = str_0.replace(',','')
        str_2 = str_1.replace('mil. €', '')
        str_2 = int(str_2)/100
#         print(str_2)
    else:
        str_2 = str_0.replace('K €', '')
        str_2 = float(str_2)/1000
#         print(str_2)
    return str_2


# In[434]:


df_clean = df_all.drop(["index"],axis="columns").set_index("#")
df_clean['Market value'] = df_clean['Market value'].apply(convert_int)
df_clean['Age'] = pd.to_numeric(df_clean['Age'])


# In[522]:


# fig, ax = plt.subplots()

categories = np.unique(df_clean['Age'])
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')	

for i, category in enumerate(categories):	
    plt.scatter('Position' ,'Market value',
                data=df_clean.loc[df_clean['Age']==category, :],
                s=20, cmap=colors[i], label=str(category))	

plt.gca().set(xlabel='Position', ylabel='Value/mil. €')	
plt.xticks(rotation=45 ,fontsize=9) 
plt.yticks(fontsize=12)
plt.title("Top250 Player Value vs Position", fontsize=22)	
plt.legend(fontsize=10)
plt.savefig("fig1.png")
plt.show() 


# In[523]:


# plt.scatter(df.close, df.volume, marker='*')
# plt.colorbar()

##top25 Player
categories = np.unique(df_clean['Age'][:25])
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')	
for i, category in enumerate(categories):	
    plt.scatter('Position' ,'Market value',
                data=df_clean[:25].loc[df_clean['Age']==category, :],
                cmap=colors[i], label=str(category))
    
plt.title("Top 25 Player Value vs Position")
plt.xlabel('Position')
plt.ylabel('Value/mil. €')
plt.legend(fontsize=10)    	
plt.show()


# In[420]:




plt.figure(figsize=(18, 10), dpi= 80, facecolor='w', edgecolor='k')	
plt.bar(df_clean.Player[:10], df_clean['Market value'][:10], width=0.5)
plt.xlabel('Player Name')
plt.ylabel('Market Value')
plt.title('Top 10 Player Value')
plt.show()


# In[518]:


df_asian_clean = df_asian.drop(["index"],axis="columns").set_index("#")
df_asian_clean['Market value'] = df_asian_clean['Market value'].apply(convert_int)
df_asian_clean['Age'] = pd.to_numeric(df_asian_clean['Age'])


# In[536]:


categories = np.unique(df_asian_clean['Age'])
colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

plt.figure(figsize=(12, 10), dpi= 80, facecolor='w', edgecolor='k')	

for i, category in enumerate(categories):	
    plt.scatter('Nat.' ,'Market value',
                data=df_asian_clean.loc[df_asian_clean['Age']==category, :],
                cmap=colors[i], label=str(category))	

plt.gca().set(xlabel='Nation', ylabel='Value/mil. €')	
plt.xticks(rotation=45 ,fontsize=9) 
plt.yticks([0,5,10,20,30,50,80],fontsize=12)
plt.title("Top Player in CHN, JPN, KOR", fontsize=22)	
plt.legend(fontsize=10)
plt.show() 


# In[571]:


plt.scatter(df_asian_clean.Age[:5], df_asian_clean['Market value'][:5])
plt.xlabel('Age')
plt.ylabel('Market Value')
plt.ylim(0,80)
plt.title('Top 5 Player in JPN')
# plt.legend()
plt.show()


# In[581]:


plt.scatter(df_asian_clean.Age[25:30], df_asian_clean['Market value'][25:30], c='r')
plt.xlabel('Age')
plt.ylabel('Market Value')
plt.ylim(0,80)
plt.title('Top 5 Player in CHN')
plt.show()


# In[582]:


plt.scatter(df_asian_clean.Age[50:55], df_asian_clean['Market value'][50:55], c = 'g')
plt.xlabel('Age')
plt.ylabel('Market Value')
plt.ylim(0,85)
plt.title('Top 5 Player in KOR')
plt.show()


# In[ ]:




