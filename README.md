# %% [markdown]
# **TWEETER SENTIMENT**
![Apple-Google-app-review-teams-scrutinise-Musks-Twitter](https://github.com/Chemami12/Tweeter-Sentiment-NLP/assets/132896105/9e66c61b-6f7c-40c1-9add-e5f911bf6586)
**TABLE OF CONTENT

# %% [markdown]
# 1. **BUSINESS UNDERSTANDING**
#    
# 1.1 **OVERVIEW**
# 
# In our modern, technology-driven society, social media platforms, especially Twitter, have become a central hub for expressing thoughts, emotions, and viewpoints. Fortunately, advancements in technology have provided us with the tools to analyze and assess these opinions. Through the application of machine learning and natural language processing, we have a method to examine a collection of text and determine its sentiment.
# According to a report from statista.com, as of December 2022, Twitter boasted an impressive user base of over 368 million monthly active users worldwide. Handling such a vast volume of tweets manually would be an almost impossible and incredibly costly endeavor. However, with the right computational resources and a well-structured dataset for training and testing machine learning models, we can swiftly analyze thousands of tweets, gaining valuable insights into public sentiment on various brands and products.
# Understanding how a consumer perceives a brand can offer valuable insight into their buying habits, which in return, can significantly impact the financial success of the company behind the brand.
# Apple offers a diverse range of products, including hardware devices, software and services that are diverse.Their products range from the iconic iPhone and Mac computers to wearables like the Apple watch and a suite of services that are convinient. This has made Apple company to be a global technology leder, synonymous with innovation, design, excellence and pushing boundaries in the world of consumer electronics.
# On the other hand, Google is a tech giant that has delved into enhancing and organising the worlds information, virtually every aspect of our digital lives. Google has evolved into a multifaceted company with a wide range of products and services that span to search, cloud computing and artificial intelligence from its inception as a search engine. Some of its application are Gmail, google maps and android operating systems.
# The project aims to build a model that rates the sentiment of tweets based on its content for Apple and Google products .
# 
# 1.2 **PROBLEM STATEMENT**
# 
# The current systems that the platform employs does not provide rating of the tweets based on their content by users. This presents a challange to Apple and Google on how to gauge their customers satisfaction of their products and categorize opinions in real-time. The new system aims to bypass these isues and gather insight into real-time customer sentiment on their products.
# Chemami Ent. is an establishment that focuses on analyzing customer feedbacks on products by various brands. The feedback generated gives an opportunity to brands to learn what makes customers happy or un unhappy about their products, so that they can tailor products to meet their customers' needs.
# 
# 1.3 **OBJECTIVE**
# 
# To build a model that can rate sentiment of a tweet based on its content of Apple and Google products.
# 
# 
# 2. **DATA UNDERSTANDING**
# 
# The dataset was sourced from CrowdFlower via https://data.world/crowdflower/brands-and-product-emotionsis from the year 2013 and has 8,721 tweets. There are three features in this dataset:
# 
# i.   Tweet text
# 
# ii.  Where the emotion in the tweet is directed at.
# 
# iii. What emotion is directed at a brand or product.
# 
# 3. **DATA PREPARATION**
# 
# 3.1 **Importing relevant libraries**
# 
# To initiate the analysis, we import the required libraries. These libraries offer essential functionalities for data manipulation, visualization, and statistical analysis, providing the groundwork for our project.
# 
# 

# %%
!pip install nltk
# import important modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
import re
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


!pip install wordcloud
import matplotlib as mpl
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# %% [markdown]
# 3.1 **Loading the Dataset**

# %%
df = pd.read_csv(r'c:\Users\Margaret Mitey\OneDrive\Documents\judge_1377884607_tweet_product_company.csv')

# %% [markdown]
# 3.2 **Previewing the Dataset**

# %%
df.head(10)

# %%
df.shape

# %%
df.columns

# %% [markdown]
# 3.3 **Cleaning the Dataset**
# 
# To make it easier to read and analyse the data, we rename the columns and simplify the sentiment column. Removing duplicates, I can't tell sentiment and missing values. Also to ensure that the Product column is correctly aligned to the Tweet column based on its content

# %%
#Renaming the columns
df = df.rename(columns = {'tweet_text': 'Tweet', 
                         'emotion_in_tweet_is_directed_at': 'Product', 
                         'is_there_an_emotion_directed_at_a_brand_or_product': 'Sentiment'})
df.head()

# %%
#Simplify sentiment labels 
dict_sent = {'No emotion toward brand or product':"No emotion", 
             'Positive emotion':'Positive emotion',
             'Negative emotion':'Negative emotion',
             "I can't tell": "I can't tell"}
df['Sentiment'] = df['Sentiment'].map(dict_sent)

# %%
df["Sentiment"].value_counts()

# %%
# check missing values in data
df.isnull().sum()

# %%
# Print the value count of product and sentiment column
print(df['Product'].value_counts())
print(("-"*40))
print(df['Sentiment'].value_counts())

# %%
# Check for duplicated rows
print('Total duplicated rows')
print(df.duplicated().sum())
print(("-"*40))

# Check for null values
print('Total null values')
print(df.isna().sum())

# %%
#Dropping duplicates
df.drop_duplicates(inplace=True)
df

# %%
#drop any NaN in the Tweet column
df.dropna(subset=['Tweet', 'Product'], inplace=True)
df

# %%
# check missing values in data
df.isnull().sum()

# %%
#Fills NaN's with "Undefined" to help with data visualizations
df['Product'].fillna("Undefined", inplace = True)
df

# %%
df = df[df['Sentiment'] != "I can't tell"]

# %%
# Print the value count of product and sentiment column
print(df['Product'].value_counts())
print(("-"*40))
print(df['Sentiment'].value_counts())

# %% [markdown]
# Cleaning and preprocessing the Product column of the DataFrame based on keywords found in the Tweet column to assign proper product labels. This is to ensure that product category is correctly labeled in the dataset.

# %%
df["Product"].fillna("none", inplace = True)
df_none = df.loc[df["Product"] == 'none']

apple_condition = (df['Product'] == "iPad") | (df['Product'] == "iPad or iPhone App") | (df['Product'] == "iPhone") | (df['Product'] == "Other Apple product or service")
google_condition = (df['Product'] == "Other Google product or service") | (df['Product'] == "Android App") | (df['Product'] == "Android") 
df.loc[apple_condition,'Product'] = "Apple"
df.loc[google_condition, 'Product'] = "Google"

# apple loop
for word in ["Apple","iphone","apple","ipad","ipad2","iPad 2","iPhone","iPad"]:
    temp_df = df_none[df_none['Tweet'].str.contains(word)]
    temp_df['Product'].replace({'none': 'Apple'}, inplace=True)
    df_none = temp_df.combine_first(df_none)
# google loop
for word in ["Google","google","Android","android"]:
    temp_df = df_none[df_none['Tweet'].str.contains(word)]
    temp_df['Product'].replace({'none': 'Google'}, inplace=True)
    df_none = temp_df.combine_first(df_none)
df_1 = df_none.combine_first(df)


# %%
df_1 = df_1[df_1['Product'] != "none"]
df_1 = df_1[df_1['Sentiment'] != "I can't tell"]

# %% [markdown]
# 4. **EXPLORATORY DATA ANALYSIS**
# 
# Exploratory Data Analysis (EDA) is a critical step in the data analysis process that involves examining and visualizing data to gain insights, detect patterns, understand the underlying structure of the dataset in decision-making, and sets the stage for more advanced analyses. By conducting a thorough EDA, we can make better-informed decisions, build more accurate models, and derive actionable insights from your data.

# %%
#Labeling products that are not classified to either Apple of Google and lowring the case
def find_Product(Product, Tweet):
    if pd.notna(Product) and Product == 'Undetermined':
        if pd.notna(Tweet):
            lower_tweet = Tweet.lower()  # Make tweet lowercase
            is_google = ('google' in lower_tweet) or ('android' in lower_tweet)  
            is_apple = ('apple' in lower_tweet) or ('ip' in lower_tweet) 
            if is_google:
                return 'Google'
            elif is_apple:
                return 'Apple'
    return Product

df['Product'] = df.apply(lambda x: find_Product(x['Product'], x['Tweet']), axis=1)
df['Product'].value_counts()  


# %%
def countplot(df, col, hue=None, rotation=None):
    fig, ax = plt.subplots(figsize=(12,8))
    sns.countplot(data = df, x = col, hue = hue, order = df[col].value_counts().index)
    ax.set_xticklabels(labels = ax.get_xticklabels(), rotation= rotation, fontsize = 15)
    ax.set_xlabel(xlabel = col, fontsize = 20)
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.set_ylabel(ylabel = "Number of Tweets", fontsize = 20)
    ax.set_title(f"Number of Tweets per {col}", fontsize = 30)
    plt.show()


# %%
countplot(df, "Product")

# %% [markdown]
# Apple has the highest number of products and brands

# %%
#count plot for Sentiment across brands
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x="Product", hue="Sentiment", palette="Accent")
plt.title('Sentiment Distribution across Brands', fontsize=20)
plt.xlabel('Product', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(title='Sentiment', title_fontsize='14', fontsize='12', loc='upper right')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Apple products and brands had the higest number of positive emotion feedback.

# %%
countplot(df, "Product", hue = "Sentiment")

# %% [markdown]
# Apple products and brands had the highest number of tweets.

# %%
mpl.rcParams['figure.figsize'] = 20,8
sample_data = df['Tweet']

word_cloud =" "
for row in sample_data:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.imshow(wordcloud);

# %% [markdown]
# Word cloud visualization depicts text in such a way that, the more frequent words appear enlarged as compared to less frequent words. This gives us a little insight into how the data looks

# %% [markdown]
# DATA PREPROCESSING AND FEATURE ENGINEERING
# 
# Data preprocessing involves the systematic cleaning and transformation of data to ensure it is in a suitable format for modeling. This essential step simplifies the data, facilitating more effective analysis and the extraction of valuable insights. During preprocessing, tasks include eliminating hashtags, retweets, hyperlinks, punctuation, and non-letter or non-whitespace characters from the text data. It also includes removing stop-words, tokenization, stemming, lemmatization, scalling, encoding and converting data.

# %%
# Cleaning text

def cleaner(Tweet):
    Tweet = re.sub(r'@[A-Za-z0-9]+','', str(Tweet)) # removes @
    Tweet = re.sub(r'#', '', str(Tweet)) # removes hashtag
    Tweet = re.sub(r'RT[\s]+','', str(Tweet)) # removes RT
    Tweet = re.sub(r'https?:\/\/\S+', '', Tweet) # remove hyperlink in tweet
    Tweet = re.sub(r'[^\w\s]', '', Tweet) # removes punctuations
    #tweet = re.sub(r'[^a-zA-Z]','', Tweet)
    return Tweet
df['Tweet'] = df['Tweet'].apply(cleaner)


# %%
#Remove retweets, links and other characters still present
df['Tweet'] = df.Tweet.map(lambda x: re.sub('rt', " ", x)) # Remove rt (retweet)
df['Tweet'] = df.Tweet.map(lambda x: re.sub('link', " ", x))# Remove link
df['Tweet'] = df.Tweet.map(lambda x: re.sub('@', " ", x)) # Remove @
df['Tweet'] = df.Tweet.map(lambda x: re.sub('mention', " ", x)) # Remove mention 
df['Tweet'] = df.Tweet.map(lambda x: re.sub('�+', " ", x)) # remove � character

df.Tweet

# %%
#Tokenizaton, Stopword removal and lemmatization of the tweets.
tokenizer = RegexpTokenizer (r'\w{3,}')

# Create a list of stopwords in English
stopwords_list = stopwords.words('english')

# Create an instance of nltk's PorterStemmer with the variable name `stemmer`
stemmer = PorterStemmer()


# %%
def preprocess_text(text, tokenizer, stopwords_list, stemmer):
    # Standardize case (lowercase the text)
    text = text.lower()

    # Tokenize text using `tokenizer`
    tokens =tokenizer.tokenize(text)

    # Remove stopwords using `stopwords_list`
    tokens = [token for token in tokens if token not in stopwords_list]

    # Stem the tokenized text using `stemmer`
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Return the preprocessed text
    
    return stemmed_tokens
preprocess_text("This is an example sentence for preprocessing.", tokenizer, stopwords_list, stemmer)

# %%
#Lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize_and_tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]

# %%

#Preprocessing the entire dataset
text_data = df['Tweet'].apply(lambda x: preprocess_text(x, tokenizer, stopwords_list, stemmer))
text_data


# %% [markdown]
# Finally, models require the target variable to be of integer type, so let's assign 0 to No emotion, 1 to Negative emotion toward brand or product and 2 to Positive emotion.

# %%
#Reassigning sentiment values to integers
df_copy = df.copy()
df_copy['Sentiment'] = df_copy['Sentiment'].replace({'Positive emotion': 2, 'Negative emotion': 1, 'No emotion toward brand or product': 0})
df = df_copy


# %% [markdown]
# MODELING AND EVALUATION
# 
# This step of the project will include the following steps:
# 
# Specifying X and y.
# 
# Spliting the data into training and testing data.
# 
# Vectorizing the data using TF-IDF vectorizer.
# 
# Choosing a baseline model.
# 
# Tuning the selected baseline model.
# 
# Our target variable will be the 'Sentiment' column and our feature variable will be the 'Tweet' column. Once specified we will proceed to split our data with the test size set at 20%.

# %%
#Reassigning sentiment values to integers
df_copy = df.copy()
df_copy['Sentiment'] = df_copy['Sentiment'].replace({'Positive emotion': 2, 'Negative emotion': 1, 'No emotion': 0})
df = df_copy

# %%
#Specifying x and y
X = df['Tweet']  
y = df['Sentiment']  


# %% [markdown]
# Splitting the data allows evaluation of the performance of model. This is by training the model on one subset (the training set) and then assess how well it generalizes to new, unseen data from another subset (the testing set). 

# %%
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Vectorization is a way to convert the texts into numerical vectors, making it suitable for machine learning algorithms.

# %%
#Vectorizing with TF-IDF

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# %% [markdown]
# **MODELING**
# 
# Baseline model: Multinomial Naives Bayes
# 
# It is particularly best suited for text-data that are represented as tokens and best starting point sentiment analysis due to its simplicity and performance.

# %%
#Choosing a baseline model
baseline_model = MultinomialNB()
baseline_model.fit(X_train_tfidf, y_train)

# %%
#Tuning the baseline model
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Add more alpha values if needed
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_


# %%
#Evaluating performance of the model

# Make predictions
y_pred = best_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# %%
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# %% [markdown]
# The model achieved an overall accuracy of approximately 84%, which means it correctly predicted the sentiment labels for 84% of the samples in the test dataset. It also performed well in identifying positive emotion tweets, achieving high precision of 84% and recall of 99% . However, it struggles with negative and neutral sentiment tweets, with low precision and recall for these classes.
# It can also be noted that the model performs poorly in identifying and classifying No emotion tweets. Further model tuning and feature engineering may help improve its performance in this regard.

# %% [markdown]
# Random Forest Classifier
# 
# To check if there is improvement of our analysis.

# %%
# Initialize and train the Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_tfidf, y_train)

# %%
#Make prediction
y_pred = random_forest.predict(X_test_tfidf)


# %%
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)

# %%
# Replace these with your actual results
conf_matrix = [[929, 8, 115], [62, 21, 20], [315, 0, 240]]
labels = ['0', '1', '2']  # Labels for the classes

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust the font scale for better readability
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False, xticklabels=labels, yticklabels=labels)

# Set axis labels and title
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')

# Display the plot
plt.show()


# %% [markdown]
# This model has an overall accuracy of 85%, which means that it correctly predicts the sentiment of tweets approximately 85% of the time. This is a slight improvement from the baseline model. The model performs well in identifying tweets in positive sentiment but struggles with class Negative sentiment. The precision for positive sentiments is 86%, recall of 99% and f1-score of 92%.
# The model achieves an overall accuracy of 85%, indicating that it correctly predicts the sentiment of tweets for the majority of cases. For No Emotion, the precision and recall are both low at 0.00, indicating that the model struggles to correctly identify tweets with no emotion. For Negative Emotion, precision is 0.81, indicating that when the model predicts negative emotion, it is correct 81% of the time. However, recall is relatively low at 29% and for Positive Emotion, the precision is 0.86, indicating that the model correctly predicts positive emotion 86% of the time. Recall is high at 99%, meaning that the model effectively captures positive sentiment tweets.

# %% [markdown]
# Support Vector Machine
# 
# Used for multi-class sentiment analysis

# %%
#Initializing SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

#Making Predictions
y_pred = svm_classifier.predict(X_test_tfidf)

#Evaluating Performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)

# %%
confusion_mat = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues_r', cbar=False,
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# %% [markdown]
# The model achieves an overall accuracy of 86%, indicating that it correctly predicts the sentiment of tweets for the majority of cases. Precision and recall are both low at 0.00 for no emotion, indicating that the model struggles to correctly identify tweets with no emotion. The F1-Score is also low. For Negative Emotion,precision is 0.85, indicating that when the model predicts negative emotion, it is correct 85% of the time. However, recall is relatively low at 33%. For Positive Emotion, precision is 0.86, indicating that the model correctly predicts positive emotion 86% of the time. Recall high at 99%, meaning that the model effectively captures positive sentiment tweets.

# %% [markdown]
# Hyperparameter Tuning
# 
# To further tune our model for optimum performance.

# %%
# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 'auto']
}

# Create the SVM model
svm = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)  # Use TF-IDF features for text vectorization

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_svm = SVC(**best_params)
final_svm.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = final_svm.predict(X_test_tfidf)

# Evaluate the final model on the test set
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Final Model Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)


# %% [markdown]
# The model achieves an accuracy of 87%, indicating that it correctly predicts the sentiment of tweets approximately 87% of the time. For No Emotion, precision is 1.00, indicating that when the model predicts no emotion, it is almost always correct. However, recall is low, suggesting that it misses many tweets with no emotion. For Negative Emotion, precision is 0.79, indicating that the model correctly predicts negative emotion 79% of the time. Recall is moderate at 42%. For Positive Emotion, precision is 0.88, indicating that the model correctly predicts positive emotion 88% of the time. Recall is high at 98%.
# 

# %% [markdown]
# **CONCLUSION**
# 
# 
# The goal of this project was to build a model that could rate the sentiment of tweets based on its content for Apple and Google products . In order to do so, several multiple classification models were tested and identified. From the results of the various models applied for analysis, we can conclude that the SVM model with hyperparameter tuning, lead to performance improvement.The model achieved the highest accuracy of 87%  and f1-score of 93% that takes into account both precision and recall among the models tested. It performed well in identifying tweets with positive emotion but struggled with No emotion. This model will allow stakeholder to identify the priority users to target with (negative and neutral sentiment) feedback and target their ads accordingly.

# %% [markdown]
# **RECOMMENDATIONS**
# 
# ChemamiEnt would advise the stakeholder to focus their efforts on engaging with Twitter users who have expressed negative sentiment towards Apple and Google products. It's crucial to address any concerns or negative experiences these users may have had, as retaining existing customers is as vital as attracting new ones. 
# 
# Additionally, those with neutral views toward Apple and Google products should not be overlooked, as they represent a potential customer base that can be further cultivated and present an opportunity for growth, thus expanding the customer base.
# 
# The machine learning models employed in our sentiment analysis can benefit from ongoing refinement and optimization. Consistent updates, thorough model assessments, and active engagement with users allow maintainance of accuracy and applicability of our predictions.


