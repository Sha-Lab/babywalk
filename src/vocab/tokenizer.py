import re
import string
import sys
import numpy as np

# padding, unknown word, end of sentence
BASE_VOCAB = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']
VOCAB_PAD_IDX = BASE_VOCAB.index('<PAD>')
VOCAB_UNK_IDX = BASE_VOCAB.index('<UNK>')
VOCAB_EOS_IDX = BASE_VOCAB.index('<EOS>')


def read_vocab(path):
  with open(path) as f:
    vocab = [word.strip() for word in f.readlines()]
  return vocab


class Tokenizer(object):
  ''' Class to tokenize and encode a sentence. '''
  # Split on any non-alphanumeric character
  SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
  
  def __init__(self, vocab=None, no_glove=False):
    self.vocab = vocab
    self.word_to_index = {}
    
    if no_glove:
      self.vocab_bos_idx = len(vocab)
      self.add_word('<BOS>', len(vocab))
    else:
      self.vocab_bos_idx = BASE_VOCAB.index('<BOS>')
    if vocab:
      for i, word in enumerate(vocab):
        self.word_to_index[word] = i
  
  def add_word(self, word, place):
    assert word not in self.word_to_index
    self.vocab.insert(place, word)
  
  @staticmethod
  def split_sentence(sentence):
    ''' Break sentence into a list of words and punctuation '''
    toks = []
    for word in [s.strip().lower() for s in
                 Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip())
                 if len(s.strip()) > 0]:
      # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
      if all(c in string.punctuation for c in word) \
        and not all(c in '.' for c in word):
        toks += list(word)
      else:
        toks.append(word)
    return toks
  
  def encode_sentence(self, sentence):
    if len(self.word_to_index) == 0:
      sys.exit('Tokenizer has no vocab')
    encoding = []
    for word in Tokenizer.split_sentence(sentence):
      if word in self.word_to_index:
        encoding.append(self.word_to_index[word])
      else:
        encoding.append(VOCAB_UNK_IDX)
    arr = np.array(encoding)
    return arr, len(encoding)
  
  def decode_sentence(self, encoding, break_on_eos=False, join=True):
    sentence = []
    for ix in encoding:
      if ix == (VOCAB_EOS_IDX if break_on_eos else VOCAB_PAD_IDX):
        break
      else:
        sentence.append(self.vocab[ix])
    if join:
      return " ".join(sentence)
    return sentence
