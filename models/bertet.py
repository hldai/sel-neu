import tensorflow as tf
from models import bertmodel


def get_dataset(data_file, batch_size, is_train, seq_length):
    dataset = tf.data.TFRecordDataset(data_file)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_len": tf.FixedLenFeature([1], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    if is_train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=100)

    drop_remainder = True if is_train else False
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return dataset


class BertET:
    def __init__(self, bert_config, seq_length, learning_rate, dropout, train_batch_size, eval_batch_size=8,
                 load_model_file=None):
        self.input_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.input_mask = tf.placeholder(tf.int32, [None, seq_length])
        self.segment_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.hidden_dropout = tf.placeholder(tf.float32, shape=[], name="hidden_dropout_prob")
        self.attention_dropout = tf.placeholder(
            tf.float32, shape=[], name="attention_probs_dropout_prob")

        self.seq_len = seq_length
        self.lr = learning_rate
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        model = bertmodel.BertModel(
            config=bert_config,
            input_ids=self.input_ids,
            hidden_dropout_prob=self.hidden_dropout,
            attention_probs_dropout_prob=self.attention_dropout,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        self.output_layer = model.get_sequence_output()
        self.__init_session(load_model_file)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def train(self, tfrec_file):
        dataset = get_dataset(tfrec_file, self.train_batch_size, True, self.seq_len)
        next_example = dataset.make_one_shot_iterator().get_next()
        for i in range(5):
            features = self.sess.run(next_example)
            output = self.sess.run(self.output_layer, feed_dict={
                self.input_ids: features["input_ids"], self.input_mask: features["input_mask"],
                self.segment_ids: features["segment_ids"], self.hidden_dropout: self.dropout,
                self.attention_dropout: self.dropout})
            print(output.shape)
