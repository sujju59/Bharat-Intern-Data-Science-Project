# Bharat-Intern-Data-Science-Project

Bharat Intern Data Science Internship Project -Sentiment analysis,NLP,OPEN MINING



import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model\_selection import train\_test\_split

from sklearn.feature\_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear\_model import LogisticRegression

from sklearn.metrics import accuracy\_score, classification\_report, confusion\_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import LatentDirichletAllocation

from wordcloud import WordCloud

data = pd.read\_csv("/content/review.csv", on\_bad\_lines='skip')

data = data\[\['review\_text', 'rating', 'date']]

data.dropna(inplace=True)



def rating\_to\_sentiment(r):

&nbsp;   if r <= 2:

&nbsp;       return "Negative"

&nbsp;   elif r == 3:

&nbsp;       return "Neutral"

&nbsp;   else:

&nbsp;       return "Positive"



data\["sentiment"] = data\["rating"].apply(rating\_to\_sentiment)



def clean\_text(text):

&nbsp;   text = str(text).lower()

&nbsp;   text = re.sub(r"http\\S+", "", text)

&nbsp;   text = re.sub(r"\[^a-zA-Z\\s]", "", text)

&nbsp;   text = re.sub(r"\\s+", " ", text)

&nbsp;   return text.strip()



data\["clean\_text"] = data\["review\_text"].apply(clean\_text)



print("Sentiment Distribution:\\n")

print(data\["sentiment"].value\_counts())



sns.countplot(x="sentiment", data=data)

plt.title("Sentiment Distribution")

plt.show()



all\_text = " ".join(data\["clean\_text"])

wc = WordCloud(width=900, height=400, background\_color="white").generate(all\_text)



plt.figure(figsize=(12,5))

plt.imshow(wc)

plt.axis("off")

plt.title("Most Frequent Words")

plt.show()



tfidf = TfidfVectorizer(stop\_words="english", max\_features=4000)

X = tfidf.fit\_transform(data\["clean\_text"])



encoder = LabelEncoder()

y = encoder.fit\_transform(data\["sentiment"])



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&nbsp;   X, y, test\_size=0.2, random\_state=42

)



model = LogisticRegression(max\_iter=1000)

model.fit(X\_train, y\_train)



pred = model.predict(X\_test)



print("\\nAccuracy:", accuracy\_score(y\_test, pred))

print("\\nClassification Report:\\n")

print(classification\_report(y\_test, pred))



cm = confusion\_matrix(y\_test, pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



count\_vectorizer = CountVectorizer(stop\_words="english", max\_df=0.95, min\_df=2)

dtm = count\_vectorizer.fit\_transform(data\["clean\_text"])



lda = LatentDirichletAllocation(n\_components=3, random\_state=42)

lda.fit(dtm)



print("\\nTop Words in Each Topic:\\n")

feature\_names = count\_vectorizer.get\_feature\_names\_out()



for index, topic in enumerate(lda.components\_):

&nbsp;   print(f"Topic {index + 1}:")

&nbsp;   print(\[feature\_names\[i] for i in topic.argsort()\[-8:]])

&nbsp;   print()



data\["date"] = pd.to\_datetime(data\["date"])

data\["month"] = data\["date"].dt.to\_period("M")



trend = data.groupby(\["month", "sentiment"]).size().unstack()

trend.plot(figsize=(10,5))

plt.title("Sentiment Trend Over Time")

plt.xlabel("Month")

plt.ylabel("Number of Reviews")

plt.show()

