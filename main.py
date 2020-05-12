import yake
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import re
import pickle

class TagExtraction:

    def __init__(self):
        print("Tag Extraction")

    
    def load_multiple_csv(self, path, column):
        df = pd.concat([pd.read_csv(f"{path}/{f}") for f in os.listdir(f"{path}/")], ignore_index=True)
        return df[column]


    def get_complexe_tag(self, document):
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=50)
        keywords = kw_extractor.extract_keywords(document)

        complexe_tag = []
        for keyword in keywords:
            if keyword[-1] >= 0.1:
                break
            complexe_tag.append(keyword[0])
        return complexe_tag


    def get_simple_tag(self, method, complexe_tags, n_topics, n_words):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
        tfidf = tf_idf_vectorizer.fit_transform(complexe_tags)
        
        decomp = None
        if method == "LDA":
            decomp = LatentDirichletAllocation(n_components=n_topics)
            decomp.fit(tfidf)
        if method == "NMF":
            decomp = NMF(n_components=n_topics)
            decomp.fit(tfidf)

        simple_tag = []

        for topic in decomp.components_:
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


    def tokenize(self, tags):
        tokenized_tags = []
        for tag in tags:
            tokenized_tags.extend(nltk.word_tokenize(tag))
        return tokenized_tags


    def save_to_file(self, path, name, data):
        with open(f'{path}/{name}.txt', 'wb') as file:
            pickle.dump(data, file)


    def load_from_file(self, path, name):
        with open(f'{path}/{name}.txt', 'rb') as file:
            data = pickle.load(file)
        return data


    def application_process(self, data, method="LDA", yake=False):
        df = pd.DataFrame(data=data)
        complexe_tags = []
        D = []
        attached_tags = []

        if yake:
            print("==== Starting complexe tag extraction ====")
            for idx, document in enumerate(df["content"]):
                print("Index :", idx)
                
                if not document or document.isspace() or re.search('[a-zA-Z]', document) == None:
                    D.append([])
                    continue
                
                tags = self.get_complexe_tag(document)
                complexe_tags.extend(tags)
                D.append(self.tokenize(tags))
            print("==========================================")
            print("==== Saving data ====")
            self.save_to_file("saved_data", "D", D)
            self.save_to_file("saved_data", "complexe_tags", complexe_tags)
            print("==========================================")
        else:
            complexe_tags = self.load_from_file("saved_data", "complexe_tags")
            D = self.load_from_file("saved_data", "D")


        print("==== Starting simple tag extraction ====")
        simple_tags = self.get_simple_tag(method, complexe_tags, 30, 30)
        print("==========================================")

        print("==== Starting list intersection ====")
        for idx, complexe_tag in enumerate(D):
            print("Index :", idx)
            attached_tags.append(self.list_intersection(complexe_tag, simple_tags))
        df["tags"] = attached_tags
        print("==========================================")

        print("==== Process finish ====")
        return df


if __name__ == "__main__":
    tag_extractor = TagExtraction()
    data = tag_extractor.load_multiple_csv("data", "content")
    df = tag_extractor.application_process(data, "LDA", False)
    df.to_csv("exports/export_LDA.csv")