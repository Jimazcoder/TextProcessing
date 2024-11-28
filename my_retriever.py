import numpy as np
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.doc_vectors = self.form_doc_vectors()

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
        for term, docs in self.index.items():
            if term in self.query_vector:
                for doc, val in docs.items():
                    self.filtered_docs.update({doc : 0})
        return self.filtered_docs
    
    def form_binary_matrix(self):
        self.filtered_docs = self.compute_tf_unique_doc_count()
        self.binary = {}
        for doc in self.filtered_docs:
            binary = {}
            for term in self.query_vector:
                row = self.index.get(term)
                if row is not None and doc in row:
                    binary.update({term : 1})
                else:
                    binary.update({term : 0})     
            self.binary.update({doc: binary})

    def form_doc_vectors(self):
        doc_vectors = {}
        for term in self.index:
            row = self.index.get(term)
            for doc, tf in row.items():
                if doc in doc_vectors:
                    current_doc = doc_vectors.get(doc)
                    current_doc.update({term : tf})
                    doc_vectors.update({doc : current_doc})
                else:    
                    doc_vectors.update({doc : {term : tf}})
        return doc_vectors
    
    def form_tf_matrix(self):
        self.filtered_doc_vectors = {}
        for doc in self.doc_vectors:
            doc_vector = self.doc_vectors.get(doc)
            for term in self.query_vector:
                if term in doc_vector:
                    self.filtered_doc_vectors.update({doc : doc_vector})
                    break

        return self.filtered_doc_vectors
    
    #tf_idf = tf * idf
    #idf = log(no. docs / 1 + no. docs containing t)
    def form_inverted_tf_idf_matrix(self):
        self.tf_idf = {}
        N = len(self.filtered_doc_vectors)
        for doc, term_tf in self.tf.items():    #{doc1: {term1: tf, term2: tf...}...}
            term_tfidf = {}
            for term, tf in term_tf.items():
                DFt = len(self.index.get(term))
                term_tfidf.update({term : tf*np.log10(N/1 + DFt)})
            self.tf_idf.update({doc : term_tfidf})
        return self.tf_idf
    
    def calculateCosSimilarity(self):
        self.similarityMatrix = {}
        docs = {}
        if self.term_weighting == 'tfidf':
            docs = self.tf_idf
        elif self.term_weighting == 'tf':
            docs = self.tf
        elif self.term_weighting == 'binary':
            docs = self.binary
        
        for doc, frequency in docs.items():    # -> {doc: {term: tfidf, ...} ...} 
                numerator = 0
                query_denom = 0
                doc_denom = 0
                for term in self.query_vector:
                    numerator += ((frequency.get(term) or 0) * self.query_vector.get(term))
                    query_denom += np.square(self.query_vector.get(term))
                    doc_denom += np.square(frequency.get(term) or 0)
                result = numerator/(np.sqrt(query_denom)*np.sqrt(doc_denom))
                self.similarityMatrix.update({doc : result})
        return self.similarityMatrix

                               

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        self.form_query_vector(query)
        if self.term_weighting == "binary":
            self.form_binary_matrix()
        self.tf = self.form_tf_matrix()
        self.form_inverted_tf_idf_matrix()
        self.calculateCosSimilarity()
        results_list = dict(sorted(self.similarityMatrix.items(), key=lambda item: item[1], reverse=True))
        results_list_doc_ids = results_list.keys()
        return list(results_list_doc_ids)


