#!/usr/bin/env python
import json
import math
import os
import re
import sys
import pprint
import heapq
from operator import itemgetter

from PorterStemmer import PorterStemmer

class IRSystem:

  def __init__(self):
    # For holding the data - initialized in read_data()
    self.titles = []
    self.docs = []
    self.vocab = []
    # For the text pre-processing.
    self.alphanum = re.compile('[^a-zA-Z0-9]')
    self.p = PorterStemmer()


  def get_uniq_words(self):
    uniq = set()
    for doc in self.docs:
      for word in doc:
        uniq.add(word)
    return uniq


  def __read_raw_data(self, dirname):
    print "Stemming Documents..."

    titles = []
    docs = []
    os.mkdir('%s/stemmed' % dirname)
    title_pattern = re.compile('(.*) \d+\.txt')

    # make sure we're only getting the files we actually want
    filenames = []
    for filename in os.listdir('%s/raw' % dirname):
      if filename.endswith(".txt") and not filename.startswith("."):
        filenames.append(filename)

    for i, filename in enumerate(filenames):
      title = title_pattern.search(filename).group(1)
      print "  Doc %d of %d: %s" % (i+1, len(filenames), title)
      titles.append(title)
      contents = []
      f = open('%s/raw/%s' % (dirname, filename), 'r')
      of = open('%s/stemmed/%s.txt' % (dirname, title), 'w')
      for line in f:
        # make sure everything is lower case
        line = line.lower()
        # split on whitespace
        line = [xx.strip() for xx in line.split()]
        # remove non alphanumeric characters
        line = [self.alphanum.sub('', xx) for xx in line]
        # remove any words that are now empty
        line = [xx for xx in line if xx != '']
        # stem words
        line = [self.p.stem(xx) for xx in line]
        # add to the document's conents
        contents.extend(line)
        if len(line) > 0:
          of.write(" ".join(line))
          of.write('\n')
      f.close()
      of.close()
      docs.append(contents)
    return titles, docs


  def __read_stemmed_data(self, dirname):
    print "Already stemmed!"
    titles = []
    docs = []

    # make sure we're only getting the files we actually want
    filenames = []
    for filename in os.listdir('%s/stemmed' % dirname):
      if filename.endswith(".txt") and not filename.startswith("."):
        filenames.append(filename)

    if len(filenames) != 60:
      msg = "There are not 60 documents in ../data/RiderHaggard/stemmed/\n"
      msg += "Remove ../data/RiderHaggard/stemmed/ directory and re-run."
      raise Exception(msg)

    for i, filename in enumerate(filenames):
      title = filename.split('.')[0]
      titles.append(title)
      contents = []
      f = open('%s/stemmed/%s' % (dirname, filename), 'r')
      for line in f:
        # split on whitespace
        line = [xx.strip() for xx in line.split()]
        # add to the document's conents
        contents.extend(line)
      f.close()
      docs.append(contents)

    return titles, docs


  def read_data(self, dirname):
    """
    Given the location of the 'data' directory, reads in the documents to
    be indexed.
    """
    # NOTE: We cache stemmed documents for speed
    #     (i.e. write to files in new 'stemmed/' dir).

    print "Reading in documents..."
    # dict mapping file names to list of "words" (tokens)
    filenames = os.listdir(dirname)
    subdirs = os.listdir(dirname)
    if 'stemmed' in subdirs:
      titles, docs = self.__read_stemmed_data(dirname)
    else:
      titles, docs = self.__read_raw_data(dirname)

    # Sort document alphabetically by title to ensure we have the proper
    # document indices when referring to them.
    ordering = [idx for idx, title in sorted(enumerate(titles),
      key = lambda xx : xx[1])]

    self.titles = []
    self.docs = []
    numdocs = len(docs)
    for d in range(numdocs):
      self.titles.append(titles[ordering[d]])
      self.docs.append(docs[ordering[d]])

    # Get the vocabulary.
    self.vocab = [xx for xx in self.get_uniq_words()]


  ##
  # Computes and store TF-IDF values for words and documents.
  ##
    # Useful data structures:
    # * self.vocab: a list of all distinct (stemmed) words
    # * self.docs: a list of lists, where the i-th document is
    #              self.docs[i] => ['word1', 'word2', ..., 'wordN']
    # NOTE: you probably do *not* want to store a value for every
    # word-document pair, but rather just for those pairs where a
    # word actually occurs in the document.
  def compute_tfidf(self):
    print "Calculating tf-idf..."

    # Initialize an empty dict for tfidf scores
    self.tfidf = {}

    # Get the log_10 of the number of docs
    logN = math.log(len(self.docs), 10)

    for word in self.vocab:
      ##
      # Retreive index of docs containing the word and the indices of 
      # occurrence of that word in each doc.
      posting = self.inv_index[word]

      ##
      # Calculate the log_10 of the inverted document frequency, which
      # is log(# of docs / # of docs in which the doc occurs).
      idf = logN - math.log(len(posting), 10)

      ##
      # Iterate through each doc index and its corresponding list containing
      # indices of occurrence of 'word' in that doc.
      for doc_i, occurrences in posting.items():
        ##
        # If the 'tfidf' of 'word' has not yet been computed and placed
        # in the 'self.tfidf' dict, initialize an empty dict and associate
        # it with 'self.tfidf[word]'.
        if word not in self.tfidf:    self.tfidf[word] = {}

        # Calculate the log term frequency.
        tf = 1.0 + math.log(len(occurrences), 10)

        ##
        # The tfidf score for the document 'd' and the current word 'word'
        # is the product of the 'tf' and 'idf' scores.
        self.tfidf[word][doc_i] = tf * idf 

    # Calculate per-document l2 norms for use in cosine similarity
    # self.tfidf_l2norm[d] = sqrt(sum[tdidf**2])) for tdidf of all words in 
    # document number d
    tfidf_l2norm2 = {}
    for word, d_dict in self.tfidf.items():
        for d,val in d_dict.items():
            tfidf_l2norm2[d] = tfidf_l2norm2.get(d, 0.0) + val ** 2
    self.tfidf_l2norm = dict((k,math.sqrt(v)) for k,v in tfidf_l2norm2.items())   
    # ------------------------------------------------------------------

  ##
  # Returns the tf-idf weigthing for the given word (string) & document index.
  def get_tfidf(self, word, document):
    return self.tfidf[word][document]


  def get_tfidf_unstemmed(self, word, document):
    """
    This function gets the TF-IDF of an *unstemmed* word in a document.
    Stems the word and then calls get_tfidf. You should *not* need to
    change this interface, but it is necessary for submission.
    """
    word = self.p.stem(word)
    return self.get_tfidf(word, document)

  ######
  # Build an inverted, positional index to allow easy access to: 
  #   1) the documents in which a particular word is contained, and 
  #   2) for every document, the positions of that word in the document 
  ##
    # Inverted index structure:
    #     inv_index = {
    #       word1: {
    #         doc1: [positn1, positn2, ..., positnN]
    #         doc2: [positn1, positn2, ..., positnN]
    #       },
    #       word2: {
    #         doc1: [positn1, positn2, ..., positnN]
    #         doc2: [positn1, positn2, ..., positnN]
    #       }
    #     }
  ##
    # Some helpful instance variables:
    #   * self.docs = List of documents
    #   * self.titles = List of titles
  def index(self):
    print "Indexing..."

    inv_index = {}

    # Iterate through each document
    for i in range(0, len(self.docs)):
      doc = self.docs[i]
      for j, word in enumerate(doc):
        ##
        # If the word has not yet been added to 'inv_index', create
        # an empty entry for it, which will later be filled.
        if not word in inv_index:       inv_index[word] = {}

        ##
        # If the current document's index has not yet been recorded
        # in the word's dictionary, create an empty array associated
        # with that doc's id. This array will subsequently be filled
        # with the corresponding indices of the words.
        if not i in inv_index[word]:    inv_index[word][i] = []

        ##
        # Append index 'j' of this occurrence of 'word' to to the
        # array corresponding to to the doc index 'i' and this 'word'
        inv_index[word][i].append(j)

    self.inv_index = inv_index


  ######
  # Given a word, this returns the list of document indices (sorted)
  # in which the word occurs. Functions as an API to self.index.
  def get_posting(self, word):
    return list(self.inv_index[word].keys())

  ######
  # Given a word, this *stems* the word and then calls get_posting on
  # the stemmed word to get its postings list. You should *not* need
  # to change this function. It is needed for submission.
  def get_posting_unstemmed(self, word):
    word = self.p.stem(word)
    return self.get_posting(word)

  ######
  # Given a query in the form of a list of *stemmed* words, this
  # returns the list of documents in which *all* of those words occur
  # (ie an AND query).
  ##
    # Return an empty list if the query does not return any documents.
  def boolean_retrieve(self, query):
    # Initialize an array with same length as query
    postings = [0]*len(query)

    ##
    # For each word in the query, get the set of document indices
    # in which that word occurs.
    for i, word in enumerate(query):
      postings[i] = set(self.get_posting(word))

    ##
    # Find the intersection of posting lists for every word in the
    # query, and turn it into a list.
    docs = list(set.intersection(*postings))

    # Return the sorted list of docs containing every word in the query.
    return sorted(docs)  # sorted doesn't actually matter 


  ######
  # Given a query in the form of an ordered list of *stemmed* words,
  # this returns the list of documents in which *all* of those words
  # occur, and in the specified order. 
  ##
    # Return an empty list if the query does not return any documents. 
  ##
    # ---PSEUDOCODE:---
    # For each occurrence of the first word
    #   Check if 2nd word is +1 ahead
    #   Check if 3rd word is +2 ahead
    #   ...
  def phrase_retrieve(self, query):
    # The list of docs to return at the end
    docs = []

    ##
    # Filter a starting list of docs to just those that we know to
    # contain all of the words.
    bool_docs = self.boolean_retrieve(query)
    if len(query) == 0:   return bool_docs

    # Iterate through each document.
    for i, doc in enumerate(bool_docs):
      ##
        # Populate 'occurrences_in_doc' with the list of a given word's
        # occurrence indices.
      occurrences_in_doc = [0]*len(query)  # init w/ length = len(query)
      for i, word in enumerate(query):
        occurrences_in_doc[i] = self.inv_index[word][doc]

      ##
      # For each occurrence of the 0th query word, look for a full
      # occurrence of the entire phrase by iterating through each word
      # in the query and checking to see if it's at the i_next index.
      for i_start in occurrences_in_doc[0]:
        full_occurrence = True
        for steps_ahead, word in enumerate(query):
          # Don't have to check 0th word in query
          if (steps_ahead==0):   pass

          i_next = i_start + steps_ahead

          ##
          # If the next word in query isn't found at the index
          # (i_start + steps_ahead) in the doc, this occurrences is
          # not complete. Break out of loop.
          if not i_next in occurrences_in_doc[steps_ahead]:
            full_occurrence = False
            break

        if full_occurrence:
          docs.append(doc)

    # Return the sorted list of documents that contain the query phrase
    return sorted(docs)   # sorted doesn't actually matter


  ######
  # Given a query (a list of words), returns a rank-ordered list of 10
  # documents (by ID) and score for the query. The score for a given doc
  # is the cosine similarity between that doc and the query.
  def rank_retrieve(self, query):

    ####### BEGIN HELPER METHODS ##############################################
    ##
      # Iterate through each word in the query, and build up a dict containing
      # a log frequency for each distinct token.
      #   - For queries w/ no repeated words, the vector will be filled with 1.0s
      #   - First we count up the raw counts, then we replace them with the log.
    def get_log_query_counts(q):
      query_counts = {}
      # If key is not yet defined, .get() retrieves 0.0.
      for word in query:   query_counts[word] = query_counts.get(word, 0.0) + 1
      # Take the log of each count and place into the dict
      return dict((word, math.log(query_counts[word], 10) + 1.0) for word in query_counts)

    ##
      # Return score for document 'd' given the specified query. The score
      # is defined as 'cos(query_counts * tfidf_vecs/norm)'' where:
      #   - tfidf_vecs[word] = tfidf of 'word' in doc number d 
      #   - norm = sqrt(tfidf_vecs[w]**2) for all words w in doc number d
    def get_score(d):
      ##
      # Create a **dictionary** in which the keys are the words from the query
      # and their corresponding values are vectors containing the tfidf of
      # that word in doc number d.
      tfidf_vecs = {}
      for word in query_counts:
        # Retrieve the array of word's tfidf values for the given doc; if key
        # is not available, it retrieves 0.0. Place this value into tfidf_vecs.
        tfidf_vecs[word] = self.tfidf[word].get(d, 0.0)

      tfidf_sum = sum(query_counts[word] * tfidf_vecs[word] for word in tfidf_vecs)
      return tfidf_sum / self.tfidf_l2norm[d]
    ######## END HELPER METHOD ################################################

    query_counts = get_log_query_counts(query)
    # Compute scores and add to a priority queue
    scores = []
    for doc_i in range(len(self.docs)):
      scores.append( (doc_i, get_score(doc_i)) )

    # Return the top 10 scores.
    #  1. Sort the doc_i-score tuples by score
    #  2. Reverse the list so that it's greatest to least
    #  3. Return only scores 0-9 to get top 10 results.
    return sorted(scores, key=itemgetter(1), reverse=True)[0:9]


  ##
  # Given a query string, process it and return the list of lowercase,
  # alphanumeric, stemmed words in the string.
  def process_query(self, query_str):
    # make sure everything is lower case
    query = query_str.lower()
    # split on whitespace
    query = query.split()
    # remove non alphanumeric characters
    query = [self.alphanum.sub('', xx) for xx in query]
    # stem words
    query = [self.p.stem(xx) for xx in query]
    return query

  ##
  # Given a string, process and then return the list of matching documents
  # found by boolean_retrieve().
  def query_retrieve(self, query_str):
    query = self.process_query(query_str)
    return self.boolean_retrieve(query)

  def phrase_query_retrieve(self, query_str):
    """
    Given a string, process and then return the list of matching documents
    found by phrase_retrieve().
    """
    query = self.process_query(query_str)
    return self.phrase_retrieve(query)

  ##
  # Given a string, process and then return the list of the top matching
  # documents, rank-ordered.
  def query_rank(self, query_str):
    query = self.process_query(query_str)
    return self.rank_retrieve(query)


def run_tests(irsys):
  print "===== Running tests ====="

  ff = open('../data/queries.txt')
  questions = [xx.strip() for xx in ff.readlines()]
  ff.close()
  ff = open('../data/solutions.txt')
  solutions = [xx.strip() for xx in ff.readlines()]
  ff.close()

  epsilon = 1e-4
  for part in range(5):
    points = 0
    num_correct = 0
    num_total = 0

    prob = questions[part]
    soln = json.loads(solutions[part])

    if part == 0:   # inverted index test
      print "Inverted Index Test"
      words = prob.split(", ")
      for i, word in enumerate(words):
        num_total += 1
        posting = irsys.get_posting_unstemmed(word)
        if set(posting) == set(soln[i]):
          num_correct += 1

    elif part == 1:   # boolean retrieval test
      print "Boolean Retrieval Test"
      queries = prob.split(", ")
      for i, query in enumerate(queries):
        num_total += 1
        guess = irsys.query_retrieve(query)
        if set(guess) == set(soln[i]):
          num_correct += 1

    elif part == 2: # phrase query test
      print "Phrase Query Retrieval"
      queries = prob.split(", ")
      for i, query in enumerate(queries):
        num_total += 1
        guess = irsys.phrase_query_retrieve(query)
        if set(guess) == set(soln[i]):
          num_correct += 1

    elif part == 3:   # tfidf test
      print "TF-IDF Test"
      queries = prob.split("; ")
      queries = [xx.split(", ") for xx in queries]
      queries = [(xx[0], int(xx[1])) for xx in queries]
      for i, (word, doc) in enumerate(queries):
        num_total += 1
        guess = irsys.get_tfidf_unstemmed(word, doc)
        if guess >= float(soln[i]) - epsilon and \
            guess <= float(soln[i]) + epsilon:
          num_correct += 1

    elif part == 4:   # cosine similarity test
      print "Cosine Similarity Test"
      queries = prob.split(", ")
      for i, query in enumerate(queries):
        num_total += 1
        ranked = irsys.query_rank(query)
        top_rank = ranked[0]
        if top_rank[0] == soln[i][0]:
          if top_rank[1] >= float(soln[i][1]) - epsilon and \
              top_rank[1] <= float(soln[i][1]) + epsilon:
            num_correct += 1

    feedback = "%d/%d Correct. Accuracy: %f" % \
        (num_correct, num_total, float(num_correct)/num_total)
    if num_correct == num_total:
      points = 3
    elif num_correct > 0.75 * num_total:
      points = 2
    elif num_correct > 0:
      points = 1
    else:
      points = 0

    print "  Score: %d Feedback: %s" % (points, feedback)


def main(args):
  irsys = IRSystem()
  irsys.read_data('../data/RiderHaggard')
  irsys.index()
  irsys.compute_tfidf()

  if len(args) == 0:
    run_tests(irsys)
  else:
    query = " ".join(args)
    print "Best matching documents to '%s':" % query
    results = irsys.query_rank(query)
    for docId, score in results:
      print "%s: %e" % (irsys.titles[docId], score)


if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
