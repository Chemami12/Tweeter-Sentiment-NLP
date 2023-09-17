
# **TWEETER SENTIMENT**

![Apple-Google-app-review-teams-scrutinise-Musks-Twitter](https://github.com/Chemami12/Tweeter-Sentiment-NLP/assets/132896105/9e66c61b-6f7c-40c1-9add-e5f911bf6586)
**TABLE OF CONTENT
 
# **OVERVIEW**
In our modern, technology-driven society, social media platforms, especially Twitter, have become a central hub for expressing thoughts, emotions, and viewpoints. Fortunately, advancements in technology have provided us with the tools to analyze and assess these opinions. Through the application of machine learning and natural language processing, we have a method to examine a collection of text and determine its sentiment. According to a report from statista.com, as of December 2022, Twitter boasted an impressive user base of over 368 million monthly active users worldwide. Understanding how a consumer perceives a brand can offer valuable insight into their buying habits, which in return, can significantly impact the financial success of the company behind the brand.            The project aims to build a model that rates the sentiment of tweets based on its content for Apple and Google products .
# **PROBLEM STATEMENT**
The current systems that the platform employs does not provide rating of the tweets based on their content by users. This presents a challange to Apple and Google on how to gauge their customers satisfaction of their products and categorize opinions in real-time. The new system aims to bypass these isues and gather insight into real-time customer sentiment on their products. Chemami Ent. is an establishment that focuses on analyzing customer feedbacks on products by various brands. The feedback generated gives an opportunity to brands to learn what makes customers happy or un unhappy about their products, so that they can tailor products to meet their customers' needs.
# **OBJECTIVE**
To build a model that can rate sentiment of a tweet based on its content of Apple and Google products.
# **DATA**
# 
The dataset was sourced from CrowdFlower via https://data.world/crowdflower/brands-and-product-emotionsis from the year 2013 and has 8,721 tweets. There are three features in this dataset:
# 
i.   Tweet text
ii.  Where the emotion in the tweet is directed at.
iii. What emotion is directed at a brand or product.
# 
# **DATA PREPARATION**
# 
To make it easier to read and analyse the data, we rename the columns and simplify the sentiment column. Removing duplicates, I can't tell sentiment and missing values. Also to ensure that the Product column is correctly aligned to the Tweet column based on its content.                                         Data preprocessing involves the systematic cleaning and transformation of data to ensure it is in a suitable format for modeling. This essential step simplifies the data, facilitating more effective analysis and the extraction of valuable insights. During preprocessing, tasks include eliminating hashtags, retweets, hyperlinks, punctuation, and non-letter or non-whitespace characters from the text data. It also includes removing stop-words, tokenization, stemming, lemmatization, scalling, encoding and converting data. To vectorize the data set for modeling, Sci-Kit Learn’s Term Frequency, Inverse Document Frequency (TF-IDF) and CountVectorizer packages was used.

# **MODELING**
For modeling techniques, Multinomial Naïve Bayes(good for text classification) was the baseline model, Random Forest(more advanced algorithm) as the stepup model, Support Vector Machine (SVM) and SVM with hyperparameter tuning for improved performance.
# **CONCLUSION** 
The aim of this project was to build a model that could rate the sentiment of tweets based on its content for Apple and Google products . In order to do so, several multiple classification models were tested and identified. From the results of the various models applied for analysis, we can conclude that the SVM model with hyperparameter tuning, lead to performance improvement.The model achieved the highest accuracy of 87%  and f1-score of 93% that takes into account both precision and recall among the models tested. It performed well in identifying tweets with positive emotion but struggled with No emotion. This model will allow stakeholder to identify the priority users to target with (negative and neutral sentiment) feedback and target their ads accordingly.

# **RECOMMENDATIONS**
ChemamiEnt would advise the stakeholder to focus their efforts on engaging with Twitter users who have expressed negative sentiment towards Apple and Google products. It's crucial to address any concerns or negative experiences these users may have had, as retaining existing customers is as vital as attracting new ones. Additionally, those with neutral views toward Apple and Google products should not be overlooked, as they represent a potential customer base that can be further cultivated and present an opportunity for growth, thus expanding the customer base.                                            The machine learning models employed in our sentiment analysis can benefit from ongoing refinement and optimization. Consistent updates, thorough model assessments, and active engagement with users allow maintainance of accuracy and applicability of our predictions.


#
├── Data
├── Images
├── Movie Recommendation Systems.ipynb
├── notebook.pdf
├── presentation.pdf
├── README.MD
└── 

