import json
import pandas as pd

CATEGORY_OTHER = 'Other'


def read_json_objs(filename):
    objs = list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs


def load_category_mapping(category_map_file):
    with open(category_map_file, encoding='utf-8') as f:
        df = pd.read_csv(f, header=None)
        category_map_dict = {cat_from: cat_to for cat_from, cat_to in df.itertuples(index=False)}
        categories = list(df[1].drop_duplicates())
        categories.append(CATEGORY_OTHER)
        category_id_dict = {cat: i for i, cat in enumerate(categories)}
    return category_map_dict, category_id_dict
