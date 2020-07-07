# Task 4
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Create a function to clean data from unnamed columns
def cleanunnamed(data):
    print('DataFrame Before Cleaning:\n---------------------------\n',data.head())
    collist = list(data.columns)
    for c in collist:
        if 'Unnamed' in c:
            data.drop(c, inplace=True, axis=1)
    print('DataFrame After Cleaning:\n---------------------------\n', data.head())
    return data

# Create function to remove all the special characters, numbers, and unwanted spaces from our text.
def textclean(Textdata):
    documents = []
    import re
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(Textdata)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(Textdata[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)

    return documents



#Reading the Spam.csv file with enconfig as ISO-8859 because the file was decoding
df = pd.read_csv("spam.csv", encoding = "ISO-8859-1")

#call cleanunnamed function to clean the dataframe from any unammed columns
dfdataset= cleanunnamed(df)
X, y = dfdataset['Text'], dfdataset['Class']

# dataset = textclean(dfdataset['Text'])
dataset = textclean(X)

'''
Converting Text to Numbers
The following script uses the bag of words model to convert text documents into corresponding numerical features
'''
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(dataset).toarray()



#Traing and testing dataset by splitting dataset to train and test groups with test size of 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# pass the training data and training target sets to this method. Take a look at the following script:
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
# predict the sentiment for the documents in our test set
y_pred = classifier.predict(X_test)


#evaluate the performance of a classification model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
