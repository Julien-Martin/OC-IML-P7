import json
from ibm_watson import NaturalLanguageUnderstandingV1, ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
import pandas as pd
import xml.dom.minidom
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
nltk.download('punkt')

class TagExtraction:
    apiKey = None
    apiURL = None
    authenticator = None
    natural_language_understanding = None

    def __init__(self, apiKey, apiURL):
        self.apiKey = apiKey
        self.apiURL = apiURL
        self.connect_to_api()


    def connect_to_api(self):
        self.authenticator = IAMAuthenticator(self.apiKey)
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2019-07-12',
            authenticator=self.authenticator
        )
        self.natural_language_understanding.set_service_url(self.apiURL)



    def Watson_get_complexe_tag(self, document):
        try:
            response = self.natural_language_understanding.analyze(
                text=document,
                features=Features(
                    keywords=KeywordsOptions()
                ),
                language='en'
            ).get_result()
            response_data = []
            for response in response["keywords"]:
                if response["relevance"] <= 0.5:
                    break
                tag = response["text"]
                response_data.append(tag)
            return response_data
        except ApiException as ex:
            print("Method failed with status code " + str(ex.code) + ": " + ex.message)
            pass


    def LDA_get_simple_tag(self, complexe_tags, n_topics, n_words):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
        tfidf = tf_idf_vectorizer.fit_transform(complexe_tags)
        lda = LatentDirichletAllocation(n_components=n_topics)
        lda.fit(tfidf)
        simple_tag = []
        for topic in lda.components_:
            tags = [tf_idf_vectorizer.get_feature_names()[i] for i in topic.argsort()[:-n_words - 1:-1]]
            simple_tag.extend(tags)
        return list(set(simple_tag))


    def list_intersection(self, complexe_tag, simple_tag):
        result = []
        for i in complexe_tag:
            for j in simple_tag:
                if i == j and i not in result:
                    result.append(i)
        return result


    def application_process(self, data):
        df = pd.DataFrame(data=data)
        complexe_watson_tags = []
        D = []
        attached_tags = []

        print("------------------\n")
        print("Starting Watson NLU")
        for idx, document in enumerate(df["content"]):
            print("Index :", idx)
            if not document or document.isspace():
                print("empty")
                D.append([])
                continue
            watson_tags = self.Watson_get_complexe_tag(document)
            complexe_watson_tags.extend(watson_tags)
            D.append(watson_tags)
        print("\nEnding Watson NLU")
        print("------------------\n")
        print("Saving Watson NLU results")
        np.save("exports/complexe_watson_tags.npy", complexe_watson_tags)
        np.save("exports/D.npy", D)
        print("Ending saving Watson NLU results")

        print("------------------\n")
        print("Starting Simple Tag Extraction")
        simple_tags = self.LDA_get_simple_tag(complexe_watson_tags, 20, 30)
        print("\n Ending Simple Tag Extraction")

        print("------------------\n")
        print("Stating Tag Intersection")
        for idx, complexe_tag in enumerate(D):
            print("Index :", idx)
            attached_tags.append(self.list_intersection(complexe_tag, simple_tags))
        print("\nEnding Tag Intersection")
        
        df["tags"] = attached_tags
        return df

    
    def load_data_from_csv(self):
        df = pd.concat([pd.read_csv(f"data/{f}") for f in os.listdir("data/")], ignore_index=True)
        df = df["content"]
        return df


if __name__ == "__main__":
    apiKey = ""
    apiURL = ""
    tag_extraction = TagExtraction(apiKey, apiURL)
    data = tag_extraction.load_data_from_csv()
    data = data[:2]
    df = tag_extraction.application_process(data)
    print("\n Finish Process")
    df.to_csv("export2.csv")