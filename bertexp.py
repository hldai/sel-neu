import os
from models import bertmodel, bertet
import config
from utils import datautils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
learning_rate = 0.001
dropout = 0.9
train_batch_size = 32

category_map_dict, category_id_dict = datautils.load_category_mapping(config.YELP_CATEGORY_MAP_FILE)

bertet_model = bertet.BertET(
    bert_config, config.BERT_SEQ_LEN, learning_rate=learning_rate, dropout=dropout,
    train_batch_size=train_batch_size, bert_init_checkpoint=config.YELP_BERT_INIT_CHECKPOINT,
    n_categories=len(category_id_dict))
bertet_model.train(config.YELP_TRAIN_TFREC_FILE, config.YELP_VALID_TFREC_FILE, 2000)
