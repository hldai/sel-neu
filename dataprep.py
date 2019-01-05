import tensorflow as tf
import collections
from utils import tokenization, datautils
import config


def get_tfrec_example(tok_sent_text, max_seq_len, tokenizer):
    tokens = tokenizer.tokenize(tok_sent_text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len - 2)]

    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    return input_ids, input_mask, segment_ids, tokens


def __get_feature_dict(input_ids, input_mask, segment_ids, seq_len):
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["seq_len"] = create_int_feature([seq_len])
    return features


def gen_tf_records_files(vocab_file, category_map_file, mentions_file, mention_tok_span_file, valid_idxs_file,
                         tok_sents_file, max_seq_len, train_output_file, valid_output_file):
    tokenizer = tokenization.SpaceTokenizer(vocab_file)

    mention_spans = datautils.read_json_objs(mention_tok_span_file)
    valid_idxs = set(datautils.read_json_objs(valid_idxs_file)[0])

    writer_train = tf.python_io.TFRecordWriter(train_output_file)
    writer_valid = tf.python_io.TFRecordWriter(valid_output_file)
    f = open(tok_sents_file, encoding='utf-8')
    for i, tokens_str in enumerate(f):
        input_ids, input_mask, segment_ids, tokens = get_tfrec_example(tokens_str.lower(), max_seq_len, tokenizer)
        features = __get_feature_dict(input_ids, input_mask, segment_ids, len(tokens))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        if i in valid_idxs:
            writer_train.write(tf_example.SerializeToString())
        else:
            writer_valid.write(tf_example.SerializeToString())
        if i > 500:
            break
    f.close()
    writer_train.close()
    writer_valid.close()


gen_tf_records_files(
    config.BERT_VOCAB_FILE, config.YELP_CATEGORY_MAP_FILE, config.YELP_MENTIONS_FILE,
    config.YELP_MENTION_TOKEN_SPAN_FILE, config.YELP_VALID_IDXS_FILE, config.YELP_TOK_SENTS_FILE,
    config.BERT_SEQ_LEN, config.YELP_TRAIN_TFREC_FILE, config.YELP_VALID_TFREC_FILE)
