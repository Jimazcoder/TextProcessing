import numpy as np
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.form_filtered_tf_matrix()
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        self.term_index = {} 
        index = 0
        for term in self.index:
            self.doc_ids.update(self.index[term])
            self.term_index.update({index: term})
            index += 1
        return len(self.doc_ids)
    
    def form_filtered_tf_matrix(self, query):
        self.tf = {}
        for doc in self.filtered_docs:
            tf = {}
            for term in query:
                tf.update({term: next((tf for key, tf in self.index.get(term) if key == doc), 0)}) #returns tf if doc term appears in doc, else returns 0 
            self.tf.update({doc: tf})                                                              # -> {doc1: {term1: tf, term2: tf...}...}                                                           

        for term in query:
            self.inverted_tf = {}
            self.inverted_tf.update({term : dict(self.index).get(term)})
    
    def compute_tf_unique_doc_count(self):
        index = 0
        self.filtered_docs = {}
        for key, value in dict(dict(self.tf).values).items():
            self.filtered_docs.update(key, index)
            index += 1
        return len(self.filtered_docs)
    
    #tf_idf = tf * idf
    #idf = log(no. docs / 1 + no. docs containing t)
    def form_inverted_tf_idf_matrix(self, query):
        self.tf_idf = {}
        N = len(self.filtered_docs)
        for doc, term_tf in dict(self.tf):           #{doc1: {term1: tf, term2: tf...}...}
            inverted_tf = self.inverted_tf
            term_tfidf = {term : tf*(np.log(N/1+len(inverted_tf.get(term)))) for term, tf in term_tf } #multiply all tf by idf for each doc in list comprehension
            self.tf_idf.update({doc: term_tfidf})
        return self.tf_idf
    
    def calculateCosSimilarity(self, query):
        self.similarityMatrix = {}
        if self.term_weighting == 'tfidf':
            for doc in self.filtered_docs:                  
                tf_idf_row = dict(self.tf_idf).get(doc)  #tf_idf_row -> {term1: tf_idf, term2: tf_idf ...}


            


    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        return list(range(1,11))


