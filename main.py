import yake, nltk, os, re, pickle, time, random
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
import multiprocessing as mp
from benchmark import BenchMark
from tqdm import tqdm
import gc

class TagExtraction:
    """Simple tag extraction by Julien Martin

    Returns:
        Pandas.DataFrame -- Return dataframe with document content and simple tags attached
    """
    complexe_tags = []
    simple_tags = []
    attached_tags = []

    def __init__(self, useYake=True, method="LDA", highly=100, n_topics=30, n_words=30):
        """Init class function, do all process and export .csv

        Keyword Arguments:
            useYake {bool} -- Allow user to extract complexe_tags with Yake (default: {True})
            method {str} -- Allow user to choose between LDA and NMF (default: {"LDA"})
            highly {int} -- Percent of document used to generate simple tags (default: {100})
            n_topics {int} -- Number of topics used in decomposition (default: {30})
            n_words {int} -- Number of keyword in topics used in decomposition (default: {30})
        """
        print("==== TAG EXTRACTOR ====")
        random.seed(42)
        self.method = method
        self.highly = highly
        self.n_topics = n_topics
        self.n_words = n_words
        print(f"Start {self.highly}_highly_{self.n_topics}_topics_{self.n_words}_words_{self.method}")
        benchmark = BenchMark(f"{self.highly}_highly_{self.n_topics}_topics_{self.n_words}_words_{self.method}")
        print("==== LOAD DATA ====")
        self.data = self.load_multiple_csv("data", "content")
        self.dataframe = pd.DataFrame(data=self.data)

        print("==== COMPLEXE TAGS ====")
        if useYake:
            data_return = self.parallelize_process(self.dataframe, self.get_complexe_tags, 8)
            self.complexe_tags = [val for sublist in data_return for val in sublist]
            self.save_to_file("complexe_tags", self.complexe_tags)
        else:
            self.complexe_tags = self.load_from_file("complexe_tags")

        print("==== GENERATE SIMPLE TAGS ====")
        self.get_simple_tags()

        print("==== ATTACHED TAGS ====")
        data_return = self.parallelize_process(self.complexe_tags, self.get_attached_tags, 4)
        self.attached_tags = [val for sublist in data_return for val in sublist]
        self.dataframe["tags"] = self.attached_tags

        benchmark.stopBenchMark()

        print("==== EXPORT ====")
        self.dataframe.to_csv(f"exports/{self.highly}_highly_{self.n_topics}_topics_{self.n_words}_words_{self.method}.csv")
        print("=============================")
        gc.collect()


    def load_multiple_csv(self, path, column):
        """Load multiple csv from path and column

        Arguments:
            path {string} -- Use to find all csv in os.path
            column {string} -- Column in CSV that contains document text

        Returns:
            Pandas.Series -- Return Series of the imported CSV
        """
        df = pd.concat([pd.read_csv(f"{path}/{f}") for f in tqdm(os.listdir(f"{path}/"))], ignore_index=True)
        return df[column]


    def get_document_complexe_tags(self, document):
        """Use Yake to extract TOP 50 keywords in a document

        Arguments:
            document {string} -- Document used to extract keyword

        Returns:
            list -- List of all complexe tags extract by Yake
        """
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=30)
        keywords = kw_extractor.extract_keywords(document)

        complexe_tag = []
        for keyword in keywords:
            if keyword[-1] >= 0.1:
                break
            complexe_tag.append(keyword[0])
        return complexe_tag


    def get_complexe_tags(self, df):
        """Iterations on all document to extract complexe tags for each document

        Arguments:
            df {Pandas.DataFrame} -- Dataframe that contains the document text

        Returns:
            list -- List of complexe tags
        """
        complexe_tags = []
        for document in tqdm(df["content"]):
            if not document or document.isspace() or re.search('[a-zA-Z]', document) == None:
                complexe_tags.append([])
                continue
            tags = self.get_document_complexe_tags(document)
            complexe_tags.append(tags)
        return complexe_tags


    def get_simple_tags(self):
        """Use LDA or NMF to extract simple keyword from complexe keyword

        """
        sampled_list = random.sample(self.complexe_tags, int(len(self.complexe_tags) * self.highly / 100))
        complexe_tags = [val for sublist in sampled_list for val in sublist]
        print("Tfidf vectorization")
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
        tfidf = tf_idf_vectorizer.fit_transform(complexe_tags)

        decomposition = None

        print("Decomposition")
        if self.method == "LDA":
            decomposition = LatentDirichletAllocation(n_components=self.n_topics, verbose=3, n_jobs=-1)
            decomposition.fit(tfidf)
        if self.method == "NMF":
            decomposition = NMF(n_components=self.n_topics)
            decomposition.fit(tfidf)

        print("Extract simple tags")
        simple_tag = []
        for topic in tqdm(decomposition.components_):
            tags = [tf_idf_vectorizer.get_feature_names()[i] for i in topic.argsort()[:-self.n_words - 1:-1]]
            simple_tag.extend(tags)
        self.simple_tags = list(set(simple_tag))


    def get_attached_tags(self, complexe_tags):
        """Use list intersection to get the intersection between complexe tags and simple tags for each document

        Arguments:
            complexe_tags {list} -- List of all complexe tags for each document

        Returns:
            list -- List of simple tags which can be use to tag the document
        """
        attached_tags = []
        for tags in tqdm(complexe_tags):
            tokenized_tags = self.tokenize(tags)
            intersection_tags = self.list_intersection(tokenized_tags)
            attached_tags.append(intersection_tags)
        return attached_tags


    def parallelize_process(self, data, func, n_cores=4):
        """Parallelize process with Python multiprocessing to speed the process

        Arguments:
            data {any} -- Any data type that can split by numpy
            func {function} -- Callback function use by the Pooling system

        Keyword Arguments:
            n_cores {int} -- Number of cores to used (default: {4})

        Returns:
            any -- Return the data that generate by the callback function
        """
        data_split = np.array_split(data, n_cores)
        pool = Pool(n_cores)
        data_return = pool.map(func, data_split)
        pool.close()
        pool.join()
        return data_return


    def list_intersection(self, complexe_tag):
        """Get the tags which is in the intersection between complexe_tag and simple_tags list

        Arguments:
            complexe_tag {list} -- List of complexe tags for each document

        Returns:
            list -- Tag intersection list
        """
        result = []
        for i in complexe_tag:
            for j in self.simple_tags:
                if i == j and i not in result:
                    result.append(i)
        return result


    def tokenize(self, tags):
        """Tokenize the data, for example "university of cambridge" become ["university", "of", "cambridge"]

        Arguments:
            tags {list} -- List of tags

        Returns:
            list -- Tokenized list of tags
        """
        tokenized_tags = []
        for tag in tags:
            tokenized_tags.extend(nltk.word_tokenize(tag))
        return tokenized_tags


    def save_to_file(self, name, data):
        """Save data to file (.txt)

        Arguments:
            name {string} -- Name of the generated file
            data {any} -- Data to store
        """
        if os.path.isdir("saved_data"):
            with open(f'saved_data/{name}.txt', 'wb') as file:
                pickle.dump(data, file)
        else:
            os.mkdir("saved_data")
            self.save_to_file(name, data)


    def load_from_file(self, name):
        """Load data from file (.txt)

        Arguments:
            name {string} -- Name of the file that contains data

        Returns:
            any -- Data load from the file
        """
        if os.path.isdir("saved_data"):
            with open(f'saved_data/{name}.txt', 'rb') as file:
                data = pickle.load(file)
            print("Successfully load from file")
            return data
        else:
            os.mkdir("saved_data")
            self.load_from_file(name)


if __name__ == '__main__':
    """ This is use to find the best parameters for the test datasets """
    # methods = ["LDA","NMF"]
    # n_topics = [30, 50]
    # n_words = [30, 50]
    # highlies = [10, 50, 100]
    # for method in methods:
    #     for highly in highlies:
    #         for n_topic in n_topics:
    #             for n_word in n_words:
    #                 TagExtraction(useYake=False, method=method, highly=highly, n_topics=n_topic, n_words=n_word)

    """ This is how you can use the class according to the report (P7_03_rapport)"""
    TagExtraction(useYake=False, method="LDA", highly=10, n_topics=30, n_words=30)
