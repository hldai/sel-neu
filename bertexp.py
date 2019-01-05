import os
from models import bertmodel, bertet
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
learning_rate = 0.001
dropout = 0.9
train_batch_size = 32

bertet_model = bertet.BertET(
    bert_config, config.BERT_SEQ_LEN, learning_rate=learning_rate, dropout=dropout,
    train_batch_size=train_batch_size)
bertet_model.train(config.YELP_TRAIN_TFREC_FILE)
