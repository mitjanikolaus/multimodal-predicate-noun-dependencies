import base64
import itertools
import os
import pickle
import sys
from binascii import Error

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import csv
import torch

from PIL import Image as PIL_Image

SUBJECT = "Label1"
REL = "label"
OBJECT = "Label2"

BOUNDING_BOX = "bounding_box"


def bb_size(relationship):
    bb = relationship[BOUNDING_BOX]
    return bb[2] * bb[3]


def get_tuples_no_duplicates(names):
    all_tuples = [
        (a1, a2) for a1, a2 in list(itertools.product(names, names)) if a1 != a2
    ]
    tuples = []
    for (a1, a2) in all_tuples:
        if not (a2, a1) in tuples:
            tuples.append((a1, a2))
    return tuples


# Objects (Label2)

OBJECTS_TEXTURES = [
    "Wooden",
    "Plastic",
    "Transparent",
    "(made of)Leather",
    "(made of)Textile",
]
OBJECTS_TEXTURES_TUPLES = get_tuples_no_duplicates(OBJECTS_TEXTURES)

OBJECTS_INSTRUMENTS = [
    "French horn",
    "Piano",
    "Saxophone",
    "Guitar",
    "Violin",
    "Trumpet",
    "Accordion",
    "Microphone",
    "Cello",
    "Trombone",
    "Flute",
    "Drum",
    "Musical keyboard",
    "Banjo",
]

OBJECTS_VEHICLES = [
    "Car",
    "Motorcycle",
    "Bicycle",
    "Horse",
    "Roller skates",
    "Skateboard",
    "Cart",
    "Bus",
    "Wheelchair",
    "Boat",
    "Canoe",
    "Truck",
    "Train",
    "Tank",
    "Airplane",
    "Van",
]

OBJECTS_ANIMALS = ["Dog", "Cat", "Horse", "Elephant"]

OBJECTS_VERBS = [
    "Smile",
    "Cry",
    "Talk",
    "Sing",
    "Sit",
    "Walk",
    "Lay",
    "Jump",
    "Run",
    "Stand",
]

OBJECTS_FURNITURE = ["Table", "Chair", "Bench", "Bed", "Sofa bed", "Billiard table"]

OBJECTS_OTHERS = [
    "Glasses",
    "Bottle",
    "Wine glass",
    "Coffee cup",
    "Sun hat",
    "Bicycle helmet",
    "High heels",
    "Necklace",
    "Scarf",
    "Belt",
    "Swim cap",
    "Handbag",
    "Crown",
    "Football",
    "Baseball glove",
    "Baseball bat",
    "Racket",
    "Surfboard",
    "Paddle",
    "Camera",
    "Mobile phone",
    "Houseplant",
    "Coffee",
    "Tea",
    "Cocktail",
    "Juice",
    "Cake",
    "Strawberry",
    "Wine",
    "Beer",
    "Woman",
    "Man",
    "Tent",
    "Tree",
    "Girl",
    "Boy",
    "Balloon",
    "Rifle",
    "Earrings",
    "Teddy bear",
    "Doll",
    "Bicycle wheel",
    "Ski",
    "Backpack",
    "Ice cream",
    "Book",
    "Cutting board",
    "Watch",
    "Tripod",
    "Rose",
]

OBJECTS_OTHERS += (
    OBJECTS_INSTRUMENTS
    + OBJECTS_VEHICLES
    + OBJECTS_ANIMALS
    + OBJECTS_VERBS
    + OBJECTS_FURNITURE
)
OBJECTS_OTHERS_TUPLES = get_tuples_no_duplicates(OBJECTS_OTHERS)

OBJECTS_TUPLES = OBJECTS_OTHERS_TUPLES + OBJECTS_TEXTURES_TUPLES

# Nouns (Label1)
SUBJECTS_FRUITS = [
    "Orange",
    "Strawberry",
    "Lemon",
    "Apple",
    "Coconut",
]

SUBJECTS_ACCESSORIES = ["Handbag", "Backpack", "Suitcase"]

SUBJECTS_FURNITURE = ["Chair", "Table", "Sofa bed", "Bed", "Bench"]

SUBJECTS_INSTRUMENTS = ["Piano", "Guitar", "Drum", "Violin"]

SUBJECTS_ANIMALS = ["Dog", "Cat"]

SUBJECTS_OTHERS = [
    "Wine glass",
    "Cake",
    "Beer",
    "Mug",
    "Bottle",
    "Bowl",
    "Flowerpot",
    "Chopsticks",
    "Platter",
    "Ski",
    "Candle",
    "Fork",
    "Spoon",
]

SUBJECTS = (
    SUBJECTS_OTHERS
    + SUBJECTS_FURNITURE
    + SUBJECTS_FRUITS
    + SUBJECTS_ACCESSORIES
    + SUBJECTS_INSTRUMENTS
    + SUBJECTS_ANIMALS
)

SUBJECTS_GENERAL_TUPLES = get_tuples_no_duplicates(SUBJECTS)

SUBJECTS_OTHERS_TUPLES = [
    ("Man", "Woman"),
    ("Man", "Girl"),
    ("Woman", "Boy"),
    ("Girl", "Boy"),
]

SUBJECT_TUPLES = SUBJECTS_GENERAL_TUPLES + SUBJECTS_OTHERS_TUPLES

# Relationships (.label)
RELATIONSHIPS_SPATIAL = ["at", "contain", "holds", "on", "hang", "inside_of", "under"]
RELATIONSHIPS_SPATIAL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_SPATIAL)

RELATIONSHIPS_BALL = ["throw", "catch", "kick", "holds", "hits"]
RELATIONSHIPS_BALL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_BALL)

RELATIONSHIPS_OTHERS = [
    "eat",
    "drink",
    "read",
    "dance",
    "kiss",
    "skateboard",
    "surf",
    "ride",
    "hug",
    "plays",
]
RELATIONSHIPS_OTHERS_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_OTHERS)

RELATIONSHIPS = RELATIONSHIPS_SPATIAL + RELATIONSHIPS_BALL + RELATIONSHIPS_OTHERS
RELATIONSHIPS_TUPLES = (
    RELATIONSHIPS_SPATIAL_TUPLES
    + RELATIONSHIPS_BALL_TUPLES
    + RELATIONSHIPS_OTHERS_TUPLES
)

subjects_counter = pd.read_csv(
    "data/sentence-semantics/subject_occurrences.csv",
    index_col=None,
    header=None,
    names=["subject", "count"],
)
SUBJECT_NAMES = list(subjects_counter["subject"].values)

objects_counter = pd.read_csv(
    "data/sentence-semantics/obj_occurrences.csv", index_col=None, header=None, names=["obj", "count"]
)
OBJECT_NAMES = list(objects_counter["obj"].values)

SYNONYMS_LIST = [
    ["Table", "Desk", "Coffee table"],
    ["Mug", "Coffee cup"],
    ["Glasses", "Sunglasses", "Goggles"],
    ["Sun hat", "Fedora", "Cowboy hat", "Sombrero"],
    ["Bicycle helmet", "Football helmet"],
    ["High heels", "Sandal", "Boot"],
    ["Racket", "Tennis racket", "Table tennis racket"],
    ["Crown", "Tiara"],
    ["Handbag", "Briefcase"],
    ["Cart", "Golf cart"],
    ["Football", "Volleyball (Ball)", "Rugby ball", "Cricket ball", "Tennis ball"],
    ["Tree", "Palm tree"],
]

SYNONYMS = {name: [name] for name in SUBJECT_NAMES + OBJECT_NAMES}
for synonyms in SYNONYMS_LIST:
    SYNONYMS.update({item: synonyms for item in synonyms})

# Threshold for overlap of 2 bounding boxes
THRESHOLD_SAME_BOUNDING_BOX = 0.02

NOUNS = ["Woman", "Man", "Girl", "Boy"]


def transform_to_per_pair_results(results):
    # Examples are on even indices
    results_pair = results[results.index % 2 == 0].copy()
    results_pair.loc[:, "result_example"] = results_pair["result"]

    # Counterexamples are on odd indices
    results_pair["result_counterexample"] = results[results.index % 2 == 1]["result"].values

    results_pair["result"] = results_pair["result_example"] & results_pair["result_counterexample"]
    return results_pair


def simplify_noun(noun):
    if noun == "Sun hat":
        noun = "hat"
    elif noun == "Bicycle":
        noun = "bike"
    elif noun == "Bicycle helmet":
        noun = "helmet"
    return noun


def transform_label(label):
    if label in OBJECTS_VERBS:
        if label.endswith("t"):
            label += "ting"
        elif label.endswith("e"):
            label = label[:-1] + "ing"
        elif label.endswith("n"):
            label += "ning"
        else:
            label += "ing"

    else:
        label = simplify_noun(label)

        if "(made of)" in label:
            label = label.replace("(made of)", "")

    return label.lower()


def get_local_path_of_cropped_image(file_name, dir, relationship):
    long_file_name = file_name.split(".jpg")[0] + "_rel_" + relationship["id"] + ".jpg"
    return os.path.join(
        dir, long_file_name
    )


def get_local_image_path(file_name, dir, relationship=None, cropped=False):
    if cropped:
        return get_local_path_of_cropped_image(file_name, dir, relationship)
    else:
        return os.path.join(
            dir, file_name
        )


def show_image(
    img_path,
    regions_and_attributes=None,
    target_sentence=None,
    distractor_sentence=None,
    result=None,
    prob_target_match=None,
    prob_distractor_match=None,
):
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    img_1_data = PIL_Image.open(img_path)

    plt.imshow(img_1_data)

    colors = ["green", "red"]
    ax = plt.gca()
    if regions_and_attributes:
        for relationship, color in zip(regions_and_attributes, colors):
            bb = relationship[BOUNDING_BOX]
            ax.add_patch(
                Rectangle(
                    (bb[0] * img_1_data.width, bb[1] * img_1_data.height),
                    bb[2] * img_1_data.width,
                    bb[3] * img_1_data.height,
                    fill=False,
                    edgecolor=color,
                    linewidth=3,
                )
            )
            ax.text(
                bb[0] * img_1_data.width,
                bb[1] * img_1_data.height,
                f"{relationship[SUBJECT]} {relationship[REL]} {relationship[OBJECT]} ",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
            )

    plt.tick_params(labelbottom="off", labelleft="off")

    ax.axis("off")

    if result is not None:
        ax.text(
            0,
            -0.01,
            f"Target sentence: {target_sentence}\nDistractor sentence: {distractor_sentence}\n"
            f"Result: {'SUCCESS' if result else 'FAILURE'} ({prob_target_match:.3f} vs. {prob_distractor_match:.3f})\n"
            f"{img_path}",
            size=10,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    plt.show()


def show_sample(
    sample,
    images_dir,
    print_bounding_boxes=True
):
    img_1_path = get_local_image_path(sample["img_filename"], images_dir)

    if print_bounding_boxes:
        show_image(
            img_1_path,
            [sample["relationship_target"], sample["relationship_distractor"]],
            sample["sentence_target"],
            sample["sentence_distractor"],
            sample["result"],
            sample["prob_target_match"],
            sample["prob_distractor_match"]
        )

    else:
        show_image(
            img_1_path,
            None,
            sample["sentence_target"],
            sample["sentence_distractor"],
            sample["result"],
            sample["prob_target_match"],
            sample["prob_distractor_match"]
        )


def load_visual_genome_vocabs():
    file = open("data/visual_genome/objects_vocab.txt")
    objects_vocab = {}
    for i, line in enumerate(file.readlines()):
        objects_vocab[i] = line.replace("\n","")

    file = open("data/visual_genome/attributes_vocab.txt")
    attrs_vocab = {}
    for i, line in enumerate(file.readlines()):
        attrs_vocab[i] = line.replace("\n", "")

    return objects_vocab, attrs_vocab


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_visual_feats(visual_feats_file):
    data = {}
    start_time = time.time()
    print("Start to load image features from %s" % visual_feats_file)
    if visual_feats_file.endswith(".tsv"):
        objs_vocab, attrs_vocab = load_visual_genome_vocabs()
        with open(visual_feats_file) as f:
            reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
            for i, item in enumerate(reader):
                data_item = {}
                for key in ['img_h', 'img_w', 'num_boxes']:
                    data_item[key] = int(item[key])

                boxes = int(item['num_boxes'])
                decode_config = [
                    ('objects_id', (boxes, ), np.int64),
                    ('objects_conf', (boxes, ), np.float32),
                    ('attrs_id', (boxes, ), np.int64),
                    ('attrs_conf', (boxes, ), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)

                    data_item[key] = item[key].reshape(shape)
                    data_item[key].setflags(write=False)

                data_item["object_names"] = [objs_vocab[obj_id] for obj_id in item["objects_id"]]
                data_item["attrs_names"] = [attrs_vocab[attr_id] for attr_id in item["attrs_id"]]

                data[item['img_id']] = data_item

    elif visual_feats_file.endswith(".p"):
        data = pickle.load(open(visual_feats_file, "rb"))
        data = {os.path.basename(key): value for key, value in data.items()}
    else:
        raise ValueError("Unknown image feature file format: ", visual_feats_file)
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(list(data.keys())), visual_feats_file, elapsed_time))
    return data


def transform_sample(tokenizer, caption, feat_raw, num_locs, max_seq_length, add_global_imgfeat, num_boxes=36):
    image_h, image_w = feat_raw['img_h'], feat_raw['img_w']

    img_feat = np.zeros((num_boxes, 2048), dtype=np.float32)
    img_pos_feat = np.zeros((num_boxes, num_locs), dtype=np.float32)

    img_feat[:num_boxes] = feat_raw['features'].copy()

    img_pos_feat[:num_boxes, :4] = feat_raw['boxes'].copy()

    if num_locs >= 5:
        img_pos_feat[:, -1] = (
                (img_pos_feat[:, 3] - img_pos_feat[:, 1])
                * (img_pos_feat[:, 2] - img_pos_feat[:, 0])
                / (float(image_w) * float(image_h))
        )

    # Normalize the box locations (to 0 ~ 1)
    img_pos_feat[:, 0] = img_pos_feat[:, 0] / float(image_w)
    img_pos_feat[:, 1] = img_pos_feat[:, 1] / float(image_h)
    img_pos_feat[:, 2] = img_pos_feat[:, 2] / float(image_w)
    img_pos_feat[:, 3] = img_pos_feat[:, 3] / float(image_h)

    if num_locs > 5:
        img_pos_feat[:, 4] = img_pos_feat[:, 2] - img_pos_feat[:, 0]
        img_pos_feat[:, 5] = img_pos_feat[:, 3] - img_pos_feat[:, 1]

    if getattr(tokenizer, "encode", None):
        input_ids = tokenizer.encode(caption)
    else:
        tokens = tokenizer.tokenize(caption)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

    token_type_ids = [0] * len(input_ids)

    input_masks = [1] * len(input_ids)
    image_masks = [1] * num_boxes

    # padding
    max_region_length = num_boxes
    while len(image_masks) < max_region_length:
        image_masks.append(0)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_masks.append(0)
        token_type_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_masks) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(image_masks) == max_region_length

    # unsqueeze batch dimension
    batch_size = 1
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_masks = torch.tensor(input_masks).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
    img_feat = torch.tensor(img_feat).unsqueeze(0)
    img_pos_feat = torch.tensor(img_pos_feat).unsqueeze(0)
    image_masks = torch.tensor(image_masks).unsqueeze(0)

    # add global [img] feature
    if add_global_imgfeat == "first":
        g_image_feat = torch.sum(img_feat, dim=1) / num_boxes
        img_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), img_feat], axis=1)
        img_feat = torch.tensor(img_feat)

        g_loc = [0, 0, 1, 1] + [1] * (num_locs - 4)
        g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
        img_pos_feat = np.concatenate([np.expand_dims(g_image_loc, axis=1), img_pos_feat], axis=1)
        img_pos_feat = torch.tensor(img_pos_feat)

        g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
        image_masks = np.concatenate([g_image_mask, image_masks], axis=1)
        image_masks = torch.tensor(image_masks)
    else:
        NotImplementedError(f"add_global_imgfeat: {add_global_imgfeat}")

    return input_ids, input_masks, token_type_ids, img_feat, img_pos_feat, image_masks


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index
