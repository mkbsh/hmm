from corpus import Document, NPChunkCorpus, NPChunkUnlabeledCorpus
from hmm import HMM
from unittest import TestCase, main
from random import shuffle, seed
import sys

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [pred == label for x in test for pred,label in zip(classifier.classify(x),x.label)]
    if verbose:
        #print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        print("%.4f%% " % (((100.0 * sum(correct) / len(correct))) if len(correct) != 0 else 0))
    return (float(sum(correct)) / len(correct)) if len(correct) != 0 else 0

class HMMTest(TestCase):
    u"""Tests for the HMM sequence labeler."""

    def split_np_chunk_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        sentences = NPChunkCorpus('np_chunking_wsj', document_class=document_class)
        seed(1)
        shuffle(sentences)
        return (sentences[:300], sentences[8936:])

    '''
    def test_np_chunk_baseline(self):
        """predicting sequences using baseline feature"""
        train, test = self.split_np_chunk_corpus(Document)
        classifier = HMM()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)
    '''


    def split_np_chunk_unlabeled_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        sentences = NPChunkUnlabeledCorpus('np_chunking_wsj_unlabeled', document_class=document_class)
        seed(1)
        shuffle(sentences)
        return (sentences[:300], sentences[8936:])

    def test_np_chunk_unlabeled_baseline(self):
        """predicting sequences using baseline feature"""
        train, test = self.split_np_chunk_corpus(Document)
        classifier = HMM()
        classifier.train_semisupervised(train)
        self.assertGreater(accuracy(classifier, test), 0.1)


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

