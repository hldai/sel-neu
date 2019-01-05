from os.path import join
from platform import platform

env = 'Windows' if platform().startswith('Windows') else 'Linux'

if env == 'Windows':
    BERT_BASE_DIR = 'd:/data/res/bert'
    YELP_DIR = 'd:/data/yelp'
else:
    BERT_BASE_DIR = '/home/hldai/data/bert'
    YELP_DIR = '/home/hldai/data/yelp'

BERT_SEQ_LEN = 128
BERT_CONFIG_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/bert_config.json')
BERT_VOCAB_FILE = join(BERT_BASE_DIR, 'uncased_L-12_H-768_A-12/vocab.txt')

YELP_CATEGORY_MAP_FILE = join(YELP_DIR, 'misc/category-map.txt')
YELP_MENTIONS_FILE = join(YELP_DIR, 'fneudata/ua-mentions.txt')
YELP_MENTION_TOKEN_SPAN_FILE = join(YELP_DIR, 'fneudata/ua-mentions-word-idxs.txt')
YELP_VALID_IDXS_FILE = join(YELP_DIR, 'fneudata/ua-mentions-valid-idxs.txt')
YELP_TOK_SENTS_FILE = join(YELP_DIR, 'fneudata/ua-mention-tok-sents.txt')
YELP_TRAIN_TFREC_FILE = join(YELP_DIR, 'fneudata/ua-mentions-train.tfrecord')
YELP_VALID_TFREC_FILE = join(YELP_DIR, 'fneudata/ua-mentions-valid.tfrecord')
