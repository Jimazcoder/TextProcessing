import numpy as np
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()

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
    
    def form_binary_matrix(self):
        self.binary = {}
        for doc in self.filtered_docs:
            binary = {}
            for term in self.query_vector:
                row = dict(self.index).get(term)
                if row is not None and doc in dict(row):
                    binary.update({term : 1})
                else:
                    binary.update({term : 0})     
            self.binary.update({doc: binary})

    def form_filtered_tf_matrix(self):
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
        for term in self.query_vector:
            self.inverted_tf.update({term : dict(self.index).get(term)})

        return self.tf
    
    #tf_idf = tf * idf
    #idf = log(no. docs / 1 + no. docs containing t)
    def form_inverted_tf_idf_matrix(self):
        self.tf_idf = {}
        N = len(self.filtered_docs)
        for doc, term_tf in dict(self.tf).items():    #{doc1: {term1: tf, term2: tf...}...}
            term_tfidf = {}
            for term, tf in dict(term_tf).items():
                if term in self.inverted_tf:
                    DFt = len(self.inverted_tf)
                else:
                    DFt = 0
                term_tfidf.update({term : tf*np.log10(N/1 + DFt)})
            self.tf_idf.update({doc : term_tfidf})
        return self.tf_idf
    
    def calculateCosSimilarity(self):
        self.similarityMatrix = {}

        if self.term_weighting == 'tfidf':
            for doc, term_tfidf in dict(self.tf_idf).items():    # -> {doc: {term: tfidf, ...} ...} 
                numerator = 0
                query_denom = 0
                doc_denom = 0
                for term in self.query_vector:
                    numerator += (dict(term_tfidf).get(term) * self.query_vector.get(term))
                    query_denom += np.square(self.query_vector.get(term))
                    doc_denom += np.square(dict(term_tfidf).get(term))
                result = numerator/(np.sqrt(query_denom)*np.sqrt(doc_denom))
                self.similarityMatrix.update({doc : result})
            return self.similarityMatrix
        
        elif self.term_weighting == 'tf':
            for doc, term_tf in dict(self.tf).items():    # -> {doc: {term: tfidf, ...} ...} 
                numerator = 0
                query_denom = 0
                doc_denom = 0
                for term in self.query_vector:
                    numerator += (dict(term_tf).get(term) * self.query_vector.get(term))
                    query_denom += np.square(self.query_vector.get(term))
                    doc_denom += np.square(dict(term_tf).get(term))
                result = numerator/(np.sqrt(query_denom)*np.sqrt(doc_denom))
                self.similarityMatrix.update({doc : result})
            return self.similarityMatrix    

        elif self.term_weighting == 'binary':
            for doc, term_binary in dict(self.binary).items():    # -> {doc: {term: tfidf, ...} ...} 
                numerator = 0
                query_denom = 0
                doc_denom = 0
                for term in self.query_vector:
                    numerator += (dict(term_binary).get(term) * self.query_vector.get(term))
                    query_denom += np.square(self.query_vector.get(term))
                    doc_denom += np.square(dict(term_binary).get(term))
                result = numerator/(np.sqrt(query_denom)*np.sqrt(doc_denom))
                self.similarityMatrix.update({doc : result})
            return self.similarityMatrix                        

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        self.form_query_vector(query)
        self.compute_tf_unique_doc_count()
        self.form_binary_matrix()
        self.form_filtered_tf_matrix()
        self.form_inverted_tf_idf_matrix()
        self.calculateCosSimilarity()
        results_list = dict(sorted(self.similarityMatrix.items(), key=lambda item: item[1], reverse=True))
        results_list_doc_ids = results_list.keys()
        return list(results_list_doc_ids)


