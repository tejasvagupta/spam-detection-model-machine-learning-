import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv('D:\\python coding\\spam_or_not_spam.csv')


model = LogisticRegression(class_weight='balanced')  


df.dropna(inplace=True)

df['label']=df['label'].astype(int)

x_train,x_test,y_train,y_test=train_test_split(df['email'],df['label'],test_size=0.2,random_state=42)

tfidf=TfidfVectorizer(stop_words='english',max_features=5000)
x_train_tdidf=tfidf.fit_transform(x_train)
x_test_tdidf=tfidf.transform(x_test)


model.fit(x_train_tdidf,y_train)

y_pred=model.predict(x_test_tdidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

def predict_spam(text):
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

user_input = input("Enter a message: ")
print("Prediction:", predict_spam(user_input))


