import tensorflow as tf
import numpy as np
from models import bertmodel, optimization


def get_dataset(data_file, batch_size, is_train, seq_length, n_categories):
    dataset = tf.data.TFRecordDataset(data_file)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_len": tf.FixedLenFeature([1], tf.int64),
        "token_span": tf.FixedLenFeature([2], tf.int64),
        "y": tf.FixedLenFeature([n_categories], tf.int64),
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
    def __init__(self, bert_config, seq_length, learning_rate, dropout, train_batch_size, n_categories,
                 eval_batch_size=8, bert_init_checkpoint=None,
                 load_model_file=None, n_train_steps=1000, n_warmup_steps=10):
        self.n_categories = n_categories
        self.seq_len = seq_length
        self.lr = learning_rate
        self.dropout = dropout
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.hidden_size = bert_config.hidden_size

        self.input_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.input_mask = tf.placeholder(tf.int32, [None, seq_length])
        self.segment_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.left_positions = tf.placeholder(tf.int32, [None])
        self.y_ture = tf.placeholder(tf.float32, [None, self.n_categories])

        self.hidden_dropout = tf.placeholder(tf.float32, shape=[], name="hidden_dropout_prob")
        self.attention_dropout = tf.placeholder(
            tf.float32, shape=[], name="attention_probs_dropout_prob")

        model = bertmodel.BertModel(
            config=bert_config,
            input_ids=self.input_ids,
            hidden_dropout_prob=self.hidden_dropout,
            attention_probs_dropout_prob=self.attention_dropout,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        self.output_layer = model.get_sequence_output()
        output_layer_nobatch = tf.reshape(self.output_layer, [-1, self.hidden_size])
        self.et_feat = tf.gather(output_layer_nobatch, self.left_positions, axis=0)

        self.W_out = tf.get_variable(
            'W_out', [self.hidden_size, self.n_categories], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.b_out = tf.get_variable(
            "b_out", [self.n_categories], initializer=tf.zeros_initializer()
        )

        self.logits_l = tf.matmul(self.et_feat, self.W_out)
        self.logits_l = tf.nn.bias_add(self.logits_l, self.b_out)
        self.logits = tf.nn.tanh(self.logits_l)
        self.y_pred = tf.nn.softmax(self.logits, axis=-1)
        self.example_losses = -tf.reduce_sum(self.y_ture * tf.log(self.y_pred), axis=-1)
        self.loss = tf.reduce_sum(self.example_losses)

        tvars = tf.trainable_variables()
        if bert_init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = bertmodel.get_assignment_map_from_checkpoint(tvars, bert_init_checkpoint)
            print('restoring parameters from {} ...'.format(bert_init_checkpoint))
            tf.train.init_from_checkpoint(bert_init_checkpoint, assignment_map)

        self.train_op = optimization.create_optimizer(
            self.loss, learning_rate, n_train_steps, n_warmup_steps, False)

        self.__init_session(load_model_file)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def load_all_eval_feats(self, tfrec_file):
        dataset = get_dataset(tfrec_file, self.eval_batch_size, False, self.seq_len, self.n_categories)
        next_example = dataset.make_one_shot_iterator().get_next()
        feats_list = list()
        while True:
            try:
                features = self.sess.run(next_example)
                feats_list.append(features)
            except tf.errors.OutOfRangeError:
                break
        return feats_list

    def evaluate(self, feats_list):
        pred_label_cnts = dict()
        hit_cnt, total_cnt = 0, 0
        for feats in feats_list:
            left_positions = feats['token_span'][:, 0]
            left_positions = np.array([i * self.seq_len + p for i, p in enumerate(left_positions)], np.int32)
            y_pred = self.sess.run(
                self.y_pred, feed_dict={
                    self.input_ids: feats['input_ids'], self.input_mask: feats['input_mask'],
                    self.segment_ids: feats['segment_ids'],
                    self.left_positions: left_positions,
                    self.hidden_dropout: 1.0,
                    self.attention_dropout: 1.0
                })

            y_pred = np.argmax(y_pred, axis=1)
            y_true = feats['y']
            # print(y_true)
            # print(y_pred)
            total_cnt += len(y_true)
            y_hits = y_true[np.arange(len(y_true)), y_pred]
            hit_cnt += np.sum(y_hits)

            for l in y_pred:
                cnt = pred_label_cnts.get(l, 0)
                pred_label_cnts[l] = cnt + 1
        print(pred_label_cnts)
        print('hc={}, tc={}, acc={:.4f}'.format(hit_cnt, total_cnt, hit_cnt / total_cnt))

    def train(self, train_tfrec_file, valid_tfrec_file, n_steps):
        print('loading {} ...'.format(valid_tfrec_file))
        valid_feats_list = self.load_all_eval_feats(valid_tfrec_file)
        print('done.')

        dataset = get_dataset(train_tfrec_file, self.train_batch_size, True, self.seq_len, self.n_categories)
        next_example = dataset.make_one_shot_iterator().get_next()
        losses = list()
        for i in range(n_steps):
            features = self.sess.run(next_example)
            left_positions = features['token_span'][:, 0]
            left_positions = np.array([i * self.seq_len + p for i, p in enumerate(left_positions)], np.int32)
            _, logits, y_true, y_pred, W_out, loss, et_feat, logits_l = self.sess.run(
                [self.train_op, self.logits, self.y_ture, self.y_pred, self.W_out, self.loss, self.et_feat, self.logits_l],
                feed_dict={
                    self.input_ids: features["input_ids"], self.input_mask: features["input_mask"],
                    self.segment_ids: features["segment_ids"],
                    self.left_positions: left_positions,
                    self.y_ture: features['y'],
                    self.hidden_dropout: self.dropout,
                    self.attention_dropout: self.dropout})

            losses.append(loss)
            # print(et_feat)
            # print(W_out)
            # print(logits)
            # print(logits_l)
            # print()

            if (i + 1) % 200 == 0:
                print(i + 1, sum(losses))
                losses = list()
                self.evaluate(valid_feats_list)
            # print(y_true)
            # print(logits)
            # print(y_pred)
            # print(example_losses)
            # print(loss)
