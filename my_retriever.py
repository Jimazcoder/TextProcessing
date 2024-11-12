import numpy as np
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
       # self.form_filtered_tf_matrix()

    def form_query_vector(self, query):
        self.query_vector = {}
        for term in query:
            if term in self.query_vector:
                self.query_vector.update({term : (self.query_vector.get(term) + 1)})
            else:
                self.query_vector.update({term : 1})
        return self.query_vector
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        self.term_index = {} 
        index = 0
        for term in self.index:
            self.doc_ids.update(self.index[term])
            self.term_index.update({index: term})
            index += 1
        return len(self.doc_ids)
    
    def compute_tf_unique_doc_count(self):
        self.filtered_docs = {}
        for term, docs in dict(self.index).items():
            if term in self.query_vector:
                for doc, val in dict(docs).items():
                    self.filtered_docs.update({doc : 0})
        return self.filtered_docs
        
    def form_filtered_tf_matrix(self, query):
        self.tf = {}
        for doc in self.filtered_docs:
            tf = {}
            for term in self.query_vector:
                row = dict(self.index).get(term)
                if row is not None and doc in dict(row):
                    tf.update({term : dict(row).get(doc)})
                else:
                    tf.update({term : 0})     
            self.tf.update({doc: tf})                                                              # -> {doc1: {term1: tf, term2: tf...}...}                                                           

        self.inverted_tf = {}
        for term in query:
            self.inverted_tf.update({term : dict(self.index).get(term)})

        return self.tf

    
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
    
    def calculateCosSimilarity(self):
        self.similarityMatrix = {}
        if self.term_weighting == 'tfidf':
            for doc, term_tfidf in self.tf_idf:    # -> {doc: {term: tfidf, ...} ...} 
                return                          

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self):#, query):
        query = ['inform', 'packet', 'radio', 'network', 'interest', 'algorithm', 'packet', 'rout', 'deal', 'network', 'topographi', 'interest', 'hardwar', 'network']
        self.form_query_vector(query)
        self.compute_tf_unique_doc_count()
        self.form_filtered_tf_matrix(query)
        print(self.inverted_tf)
        return list(range(1,11))


