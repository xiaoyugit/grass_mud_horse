import data_io
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

#callable that provides tokens for Company, ContractType, ContractTime and Source
def my_tokenizer(s):
    return s.strip().split('?')

def my_vectorizer(type):
    box = {"text": CountVectorizer(max_features = 100, stop_words = 'english'),
            "category": CountVectorizer(tokenizer = my_tokenizer, min_df = 1)
            }
    return box[type]

def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', my_vectorizer("text")),
                ('Title-Bag of Words', 'Title', my_vectorizer("text")),
                #('LocationRaw-Bag of Words', 'LocationRaw', my_vectorizer("text")),
                ('LocationNormalized-Bag of Words', 'LocationNormalized', my_vectorizer("category")),
                #('Category-Bag of Words','Category',my_vectorizer("category")),
                ('Company-Bag of Words','Company',my_vectorizer("category")),
                ('ContractType-Bag of Words','ContractType',my_vectorizer("category")),
                ('ContractTime-Bag of Words','ContractTime',my_vectorizer("category")),
                
                ('Source-Bag of Words','SourceName',my_vectorizer("category"))]
    combined = FeatureMapper(features)
    return combined

def get_pipeline(data):
    features = feature_extractor()
    features.fit(data)
    voc_set = features.get_vocabulary()
    for k in voc_set:
        print k + ':'
        print ','.join(voc_set[k])
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=30, 
                                                verbose=2,
                                                n_jobs=3,
                                                min_samples_split=30,
                                                random_state=3465344))]
             #("classify", SVC())]
             #("classify", GradientBoostingRegressor(verbose=2,min_samples_split=10  ))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()
    
    print("Extracting features and training model")
    i = 1
    for key in train:
        print "Dealing with the %s/29 model" %i
        i += 1
        classifier = get_pipeline(train[key])
        classifier.fit(train[key], train[key]["SalaryNormalized"])
        print("Saving the classifier for %s" %key)
        data_io.save_model(classifier,key)

#    print("Saving the classifier")
#    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
