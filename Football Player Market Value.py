#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import urllib.request
from bs4 import BeautifulSoup
import numpy as np


# In[3]:


import pandas as pd
from sqlalchemy import create_engine
import time
import random
import matplotlib.pyplot as plt


# In[ ]:


player_values_url = "https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop"
url2 = "?page="

japan_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=77&yt0=Show"
china_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=34&yt0=Show"
korea_player_values_url = "/plus/0/galerie/0?ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&land_id=87&yt0=Show"

player_detail_url = "&land_id=0&ausrichtung=alle&spielerposition_id=alle&altersklasse=alle&jahrgang=0&kontinent_id=0&plus=1"

ua_headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'
}


# In[ ]:


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


# In[ ]:


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
    
    print(data)
    return data


# In[ ]:


def to_df(data):
    df = pd.DataFrame(data)
    df_dropped = df.drop([1,2,6,7],axis="columns")
    df_dropped.columns = ['#', 'Player','Position', 'Age', 'Market value','Nat.', 'Club']
    df_dropped.set_index('#')
    
    return df_dropped


# In[ ]:


def to_sql(df, db_name):
    df.set_index('#')
    engine = create_engine('sqlite:///' + db_name + '.sqlite')
    df.to_sql(db_name, con = engine, if_exists='append')
    print("saved......")


# In[ ]:


for i in range(1,11):
    
    html = getHtml(player_values_url + url2 + str(i), ua_headers)
    
    data = getData(html)
    
    df = to_df(data)
    
    print(df)
    
    to_sql(df, 'PlayerValue')

print("finish!!!")


# In[ ]:


for url in [japan_player_values_url, china_player_values_url, korea_player_values_url]:
    
    html = getHtml(player_values_url + url, ua_headers)
    
    data = getData(html)
    
    df = to_df(data)
    
    print(df)
    
    to_sql(df, 'EastAsianPlayerValue')

print("finish!!!")


# In[289]:


engine = create_engine('sqlite:///PlayerValue.sqlite')

# engine.execute("SELECT * FROM PlayerValue").fetchall()
df_all = pd.read_sql_query("SELECT * FROM PlayerValue", engine)


# In[4]:


engine1 = create_engine('sqlite:///EastAsianPlayerValue.sqlite')

# engine.execute("SELECT * FROM PlayerValue").fetchall()
df_asian = pd.read_sql_query("SELECT * FROM EastAsianPlayerValue", engine1)


# In[14]:


##update the currency symbol according the website
def convert_int(str_0):
    if '.' in str_0:
        str_1 = str_0.replace('.','')
        str_2 = str_1.replace('m', '').replace('€','')
#         str_2 = str_2.replace('€','')
        str_2 = int(str_2)/100
#         print(str_2)
    else:
        str_2 = str_0.replace('€', '').replace('k','')
        str_2 = float(str_2)/1000
#         print(str_2)
    return str_2


# In[16]:


##Deprecated data function
def convert_int_pre(str_0):
    if ',' in str_0:
        str_1 = str_0.replace(',','')
        str_2 = str_1.replace('mil. €', '')
#         str_2 = str_2.replace('€','')
        str_2 = int(str_2)/100
#         print(str_2)
    else:
        str_2 = str_0.replace('K €', '')
        str_2 = float(str_2)/1000
#         print(str_2)
    return str_2


# In[291]:


df_all.head()


# In[5]:


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


# In[ ]:


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


# In[ ]:


plt.figure(figsize=(18, 10), dpi= 80, facecolor='w', edgecolor='k')	
plt.bar(df_clean.Player[:10], df_clean['Market value'][:10], width=0.5)
plt.xlabel('Player Name')
plt.ylabel('Market Value')
plt.title('Top 10 Player Value')
plt.show()


# In[17]:


df_asian_clean = df_asian.drop(["index"],axis="columns").set_index("#")
df_asian_clean['Market value'] = df_asian_clean['Market value'].apply(convert_int_pre)
df_asian_clean['Age'] = pd.to_numeric(df_asian_clean['Age'])


# In[18]:


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


# In[19]:


plt.scatter(df_asian_clean.Age[:5], df_asian_clean['Market value'][:5],label="Japan")
plt.scatter(df_asian_clean.Age[25:30], df_asian_clean['Market value'][25:30], c='r',label="China")
plt.scatter(df_asian_clean.Age[50:55], df_asian_clean['Market value'][50:55], c = 'g',label="South Korea")

plt.xlabel('Age')
plt.ylabel('Market Value')
plt.ylim(0,85)
plt.title('Top 5 Player in JPN vs CHN vs KOR')
plt.legend()
plt.show()


# In[ ]:


def getDetailedData(html):
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
        
        ##Drop the scecond nation
        if len(alt) == 3:
            player.extend([a.attrs['alt'] for a in alt[1:3]])
        elif len(alt) == 4:
            player.extend([a.attrs['alt'] for a in alt[1:4:2]])
        else:
            print("error to drop the scecond nation")
        
        data.append(player)
    
#     print(data)
    return data


# In[ ]:


def to_detailed_df(data):
    df = pd.DataFrame(data)
    df_dropped = df.drop([1,2,6,7],axis="columns")
    df_dropped.columns = ['#', 'Player','Position', 'Age', 'Market value',
                      'Matches','Goals','Own Goals','Assists','Yellow Cards',
                      'Second Yellow Card','Red Cards','Subs on','Subs off', 'Nat.', 'Club']
    df_dropped.set_index('#')
    
    return df_dropped


# In[ ]:


###Get detail information
for i in range(1,11):
    html_detailed = getHtml(player_values_url + url2 + str(i) + player_detail_url,headers=ua_headers)
    
    data_detailed = getDetailedData(html_detailed)
    
    df_detailed = to_detailed_df(data_detailed)
    
    to_sql(df_detailed, 'PlayerValueDetailed')

print("finish!!!")


# In[20]:


from sklearn.preprocessing import normalize


# In[160]:


engine2 = create_engine('sqlite:///PlayerValueDetailed.sqlite')

# engine.execute("SELECT * FROM PlayerValue").fetchall()
df_detailed = pd.read_sql_query("SELECT * FROM PlayerValueDetailed", engine2)


# In[161]:


df_detailed.head()


# In[23]:


fifa19_path = './Downloads/data.csv'

df_fifa = pd.read_csv(fifa19_path)
df_fifa.head()


# In[188]:


# ##Augmentation from FIFA19
df_detailed['Potential'] = [95, 93, 89, 91, 91, 94, 89, 90, 92, 87, 88, 89, 85, 91, 90,
                         92, 88, 91, 93, 91, 89, 90, 90, 89, 92, 94, 89, 87, 88, 89, 87, 90,
                         85, 84, 91, 94, 88, 89, 89, 90, 90, 92, 92, 90, 88, 88, 90, 85,
                         91, 88, 86, 86, 86, 88, 86, 89, 90, 89, 87, 88, 87, 93,
                         90, 89, 87, 89, 87, 87, 92, 86, 87, 88, 87, 85, 89, 91, 88, 88,
                         90, 85, 90, 88, 83, 91, 88, 90, 90, 85, 87, 93, 88, 85, 90, 87,
                         87, 86, 87, 84, 86, 88, 83, 87, 82, 88, 87, 85, 89, 88, 84, 88, 87,
                         84, 85, 85, 86, 84, 89, 84, 86, 87, 89, 84, 82, 91, 86, 86, 84, 90,
                         80, 88, 85, 87, 87, 87, 83, 89, 85, 92, 87, 86, 79, 84, 87, 82, 85, 84,
                         89, 83, 88, 80, 89, 86, 86, 84, 87, 85, 86, 86, 86, 89, 88, 88, 85, 88,
                         82, 88, 82, 83, 88, 84, 87, 84, 83, 83, 89, 84, 87, 85, 78, 83, 86, 83,
                         89, 82, 88, 87, 89, 86, 86, 87, 86, 80, 82, 85, 86, 85, 86, 87, 88,
                         88, 85, 86, 84, 79, 88, 85, 87, 85, 84, 87, 80, 86, 85, 82, 86, 82,
                         82, 86, 82, 83, 82, 86, 86, 85, 84, 89, 87, 85, 88, 88, 87, 87, 85, 86,
                         85, 85, 84, 87, 80, 84, 84, 82, 84, 82, 83, 87, 88, 81, 86, 89]


# In[189]:


df_detailed['Overall'] = [88, 92, 88, 89, 91, 94, 86, 89, 91, 86, 72, 77, 76, 86, 89,
                         86, 86, 87, 90, 88, 80, 84, 85, 88, 89, 94, 81, 82, 78, 81, 86, 85,
                         84, 76, 82, 89, 84, 86, 87, 87, 87, 77, 83, 82, 82, 84, 86, 82,
                         86, 85, 85, 85, 86, 88, 86, 80, 85, 84, 83, 88, 84, 91,
                         90, 89, 80, 81, 78, 79, 85, 78, 83, 84, 82, 85, 85, 83, 83, 82,
                         84, 85, 84, 83, 83, 88, 88, 90, 88, 85, 82, 82, 82, 82, 89, 85,
                         78, 79, 80, 73, 80, 83, 77, 81, 80, 84, 80, 80, 82, 82, 81, 83, 84,
                         83, 82, 81, 84, 84, 88, 82, 83, 87, 89, 84, 82, 91, 86, 79, 71, 80,
                         74, 81, 77, 78, 82, 84, 80, 84, 83, 87, 85, 86, 78, 74, 77, 79, 80, 66,
                         81, 80, 77, 74, 82, 81, 81, 78, 81, 77, 81, 82, 83, 83, 83, 81, 80, 83,
                         79, 81, 82, 82, 83, 81, 87, 83, 82, 82, 89, 84, 86, 85, 77, 81, 86, 82,
                         77, 79, 75, 70, 77, 80, 75, 77, 78, 75, 77, 78, 79, 80, 77, 81, 78,
                         84, 80, 79, 77, 74, 81, 79, 83, 80, 79, 82, 78, 81, 80, 80, 83, 82,
                         82, 83, 80, 81, 82, 86, 86, 85, 84, 83, 85, 85, 88, 88, 87, 87, 80, 83,
                         74, 79, 78, 81, 79, 80, 83, 82, 84, 77, 74, 76, 73, 65, 78, 68]


# In[87]:


# df_detailed.to_csv("./PlayerMarketValuewithFIFA")


# In[190]:


df_detailed.head()


# In[191]:


# df = pd.DataFrame(data_detailed)
# df_dropped = df.drop([1,2,6,7],axis="columns")
# df_dropped.columns = ['#', 'Player','Position', 'Age', 'Market value',
#                       'Matches','Goals','Own Goals','Assists','Yellow Cards',
#                       'Second Yellow Card','Red Cards','Subs on','Subs off', 'Nat.', 'Club']
# df_dropped.set_index('#')


# In[192]:


##To numeric
df_detailed_clean = df_detailed.drop(["index"],axis="columns").set_index("#")
df_detailed_clean['Market value'] = df_detailed_clean['Market value'].apply(convert_int).apply(int)
df_detailed_clean[['Age', 'Matches', 'Goals', 'Own Goals', 'Assists','Yellow Cards',
                   'Second Yellow Card','Red Cards','Subs on','Subs off']] = df_detailed_clean[
    ['Age', 'Matches', 'Goals', 'Own Goals', 'Assists','Yellow Cards',
     'Second Yellow Card','Red Cards','Subs on','Subs off']].apply(pd.to_numeric)


# In[193]:


## Drop Unnecessary Values
# drop_cols = ['Player']
drop_cols = ['Player', 'Own Goals', 'Yellow Cards', 'Second Yellow Card',
             'Red Cards', 'Subs on', 'Subs off']

df_detailed_clean = df_detailed_clean.drop(drop_cols, axis = 1)


# In[194]:


df_detailed_clean.head()


# In[62]:


import seaborn as sns


# In[63]:


# sns.jointplot(x='Overall',y ='Market value', data = df_detailed_clean)


# In[64]:


# sns.pairplot(df_detailed_clean)


# In[65]:


# df_detailed_clean[(df_detailed_clean['Goals']+df_detailed_clean['Assists'])>20]


# In[195]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_detailed_clean.Position.drop_duplicates()) 
df_detailed_clean.Position = le.transform(df_detailed_clean.Position)

le.fit(df_detailed_clean['Nat.'].drop_duplicates()) 
df_detailed_clean['Nat.'] = le.transform(df_detailed_clean['Nat.'])

le.fit(df_detailed_clean.Club.drop_duplicates()) 
df_detailed_clean.Club = le.transform(df_detailed_clean.Club)


# In[196]:


# df_detailed_clean.Goals + df_detailed_clean.Assists


# In[197]:


df_detailed_clean['GA'] = df_detailed_clean.Goals + df_detailed_clean.Assists


# In[198]:


df_detailed_clean = df_detailed_clean.drop(['Goals','Assists'], axis = 1)


# In[199]:


df_detailed_clean.head()


# In[200]:


##Create Dummy Variable
# dummy_fields = ['Club', 'Nat.', 'Position']
# for each in dummy_fields:
#     dummies = pd.get_dummies(df_detailed_clean[each], prefix=each, drop_first=False)
#     df_dummy = pd.concat([df_detailed_clean, dummies], axis=1)

# data = df_detailed_clean.drop(['Player'],axis=1)
# data = pd.get_dummies(df_detailed_clean)

# data.head()


# In[201]:


from sklearn import preprocessing

x = df_detailed_clean.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled, columns=df_detailed_clean.columns)


# In[202]:


df_scaled.head()


# In[175]:


df_detailed_clean.corr()['Market value'].sort_values(ascending=False)


# In[176]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df_detailed_clean.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax, annot=True)


# In[177]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model


# In[249]:


##Split Data
target_fields = ['Market value']
# features, targets = df_detailed_clean.drop(target_fields, axis=1), df_detailed_clean[target_fields]
features, targets = df_detailed_clean[['Overall', 'Potential', 'GA']], df_detailed_clean[target_fields]
# features, targets = df_detailed_clean[['Age', 'Overall', 'Potential', 'GA']], df_detailed_clean[target_fields]
# features, targets = df_detailed_clean[['GA']], df_detailed_clean[target_fields]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.1)


# In[250]:


from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV

# Instantiate a lasso regressor: lasso
# lasso = Lasso(alpha=0.3,normalize=True)
lasso = LassoCV(normalize=True)

# X = df_detailed_clean.drop(target_fields, axis=1)

# Fit the regressor to the data
lasso.fit(features, targets)

# Compute and print the coefficients
lasso_coef = lasso.fit(features, targets).coef_
# print(lasso.alpha_)

print(lasso_coef)

# predicted = model.predict(features)

plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(range(len(features.columns)), lasso_coef)
plt.xticks(range(len(features.columns)), features.columns.values, rotation=60)
plt.margins(0.02)
plt.show()


# In[251]:


lasso.score(features, targets)


# In[253]:


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[255]:


lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)


# In[256]:


y_pred = lr.predict(X_test)


# In[260]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# The coefficients
print('Coef_: {} Intercept_: {}\n'.format(lr.coef_, lr.intercept_))

# The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# The mean absolute error
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('(R2) Variance score: %.2f' % r2_score(y_test, y_pred))

print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))


# In[158]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(lr, features, targets, cv=5)

print(cv_scores)

print("Average 3-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[159]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train, y_train)
y_predict_svc = svc.predict(X_test)
print('score: {}'.format(svc.score(X_test, y_test)))
print('Variance score: %.2f' % r2_score(y_test, y_predict_svc))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))

