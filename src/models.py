import os
import json
import torch
import io
from PIL import Image
import numpy as np
import pandas as pd

from src.utils import load_visual_feats, transform_sample, get_gather_index

import torch.nn.functional as F


class ModelRepresentation():
    def __init__(self, device):
        self.device = device

    def initialisation(self, model):
        model.to(self.device)
        model.eval()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in model: {pytorch_total_params}")


class CLIPModelRepresentation():
    def __init__(self, device):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.initialisation(self.model)

    def initialisation(self, model):
        model.to(self.device)
        model.eval()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in model: {pytorch_total_params}")

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, img_path):
        import clip
        image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        # target_sentence = f"a photo of {target_sentence}"
        # target_sentence = target_sentence.replace(" is ", " ")
        # distractor_sentence = f"a photo of {distractor_sentence}"
        # distractor_sentence = distractor_sentence.replace(" is ", " ")
        text = clip.tokenize([target_sentence, distractor_sentence]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs[0]


class LXMERTModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file, for_pretraining=False):
        from transformers import LxmertModel, LxmertForPreTraining, BertTokenizer
        model_class = LxmertModel
        if for_pretraining:
            model_class = LxmertForPreTraining
        super(LXMERTModelRepresentation, self).__init__(device)
        checkpoint = "unc-nlp/lxmert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.model = model_class.from_pretrained(checkpoint)
        self.initialisation(self.model)
        self.visual_feat_dict = load_visual_feats(visual_feats_file)

    def transform_visual_feats(self, visual_feats):
        obj_num = visual_feats['num_boxes']
        feats = visual_feats['features'].copy()
        boxes = visual_feats['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        img_h, img_w = visual_feats['img_h'], visual_feats['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return (torch.from_numpy(feats), torch.from_numpy(boxes))

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, visual_features_path)
        return prob_target, prob_distractor

    def get_img_sent_match_score(self, test_sentence, visual_features_path):
        visual_features_key = os.path.basename(visual_features_path)

        visual_feat_raw = self.visual_feat_dict[visual_features_key]
        visual_feats, visual_pos = self.transform_visual_feats(visual_feat_raw)
        inputs = self.tokenizer(test_sentence, return_tensors="pt")

        # decoded_sequence = self.tokenizer.decode(inputs["input_ids"][0])
        # print(decoded_sequence)

        visual_attention_mask = torch.ones(visual_feats.shape[:-1], dtype=torch.float)

        inputs['visual_attention_mask'] = visual_attention_mask.unsqueeze(dim=0)
        inputs['visual_feats'] = visual_feats.unsqueeze(dim=0)
        inputs['visual_pos'] = visual_pos.unsqueeze(dim=0)
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))

        cross_relationship_score = outputs.cross_relationship_score

        # Apply softmax
        softmaxed = F.softmax(cross_relationship_score[0], dim=0)
        # return the probability for image-sentence match:
        return softmaxed[1].item()


class VisualBERTModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file, for_pretraining=False):
        super().__init__(device)
        from transformers import VisualBertForPreTraining, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        checkpoint = "uclanlp/visualbert-nlvr2-coco-pre"
        self.model = VisualBertForPreTraining.from_pretrained(checkpoint)
        self.device = device
        self.initialisation(self.model)
        self.visual_feat_dict = load_visual_feats(visual_feats_file)

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, target_sentence, visual_features_path)
        return prob_target, prob_distractor

    def get_img_sent_match_score(self, test_sentence, correct_sentence, visual_features_path):
        visual_features_key = os.path.basename(visual_features_path)
        # Problem: VisualBERT requires two sentences to be in the input, so it can solve the task by doing
        # text-to-text matching!
        inputs = self.tokenizer(correct_sentence, test_sentence, return_tensors="pt")

        # decoded_sequence = self.tokenizer.decode(inputs["input_ids"][0])
        # print(decoded_sequence)

        visual_features = self.visual_feat_dict[visual_features_key].unsqueeze(0)

        visual_token_type_ids = torch.ones(visual_features.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_features.shape[:-1], dtype=torch.float)

        inputs.update({
            "visual_embeds": visual_features,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask
        })

        with torch.no_grad():
            outputs = self.model(**inputs)

        seq_relationship_logits = outputs.seq_relationship_logits

        # Apply softmax and return probability for match:
        # - 0 indicates sequence B is a matching pair of sequence A for the given image,
        # - 1 indicates sequence B is a random sequence w.r.t A for the given image.
        softmaxed = F.softmax(seq_relationship_logits[0], dim=0)
        return softmaxed[0].data


class UNITERModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file, offline=False):
        from transformers import BertTokenizer
        from src.UNITER.model.pretrain import UniterForPretraining
        from src.UNITER.utils.const import IMG_DIM, IMG_LABEL_DIM

        super(UNITERModelRepresentation, self).__init__(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, local_files_only=offline)

        checkpoint = "models/UNITER/uniter-base.pt"
        checkpoint = torch.load(checkpoint)
        config_file = "src/UNITER/config/uniter-base.json"
        self.model = UniterForPretraining.from_pretrained(config_file, checkpoint, img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
        self.initialisation(self.model)
        self.visual_feat_dict = load_visual_feats(visual_feats_file)

    def transform_visual_feats(self, visual_feats):
        feats = visual_feats['features'].copy()
        bboxes = visual_feats['boxes'].copy()
        image_h, image_w = visual_feats['img_h'], visual_feats['img_w']
        box_width = bboxes[:, 2] - bboxes[:, 0]
        box_height = bboxes[:, 3] - bboxes[:, 1]
        scaled_width = box_width / image_w
        scaled_height = box_height / image_h
        scaled_x = bboxes[:, 0] / image_w
        scaled_y = bboxes[:, 1] / image_h

        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]

        bb = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)

        img_feat, bb = torch.from_numpy(feats.copy()), torch.from_numpy(bb.copy())

        img_pos_feat = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        return img_feat, img_pos_feat

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, visual_features_path)
        return prob_target.item(), prob_distractor.item()

    def get_img_sent_match_score(self, test_sentence, visual_features_path):
        visual_features_key = os.path.basename(visual_features_path)
        input_ids = self.tokenizer(test_sentence, return_tensors="pt")['input_ids'].to(self.device)

        # decoded_sequence = self.tokenizer.decode(input_ids[0])
        # print(decoded_sequence)

        visual_feat_raw = self.visual_feat_dict[visual_features_key]
        img_feat, img_pos_feat = self.transform_visual_feats(visual_feat_raw)
        num_bb = img_feat.size(0)
        attention_mask = torch.ones(input_ids.size(1) + num_bb, dtype=torch.long).unsqueeze(0).to(self.device)

        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)
        img_feat, img_pos_feat = img_feat.unsqueeze(0).to(self.device), img_pos_feat.unsqueeze(0).to(self.device)
        txt_lens = [i.size(0) for i in input_ids]
        bs, max_tl = input_ids.size()
        out_size = attention_mask.size(1)
        gather_index = get_gather_index(txt_lens, [num_bb], bs, max_tl, out_size).to(self.device)

        input = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attention_mask,
                 'gather_index': gather_index,
        }

        with torch.no_grad():
            outputs = self.model(input, task="itm", compute_loss=False)[0]

        # Apply softmax
        softmaxed = F.softmax(outputs[0], dim=0)
        # return the probability for image-sentence match:
        return softmaxed[1].data


class VILTModelRepresentation(ModelRepresentation):
    def __init__(self, device, offline=False):
        from vilt.modules.vilt_module import ViLTransformerSS
        from vilt.transforms import keys_to_transforms
        from transformers import BertTokenizer

        super(VILTModelRepresentation, self).__init__(device)
        with open('models/VILT/probing_config.json', 'r') as f:
            config = json.load(f)
        self.model = ViLTransformerSS(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="uncased", local_files_only=offline)
        self.config = config
        self.transforms = keys_to_transforms(config["val_transform_keys"], size=config["image_size"])
        self.image_token_type_idx = 1
        self.initialisation(self.model)

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, img_path):
        prob_target = self.get_img_sent_match_score(target_sentence, img_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, img_path)
        return prob_target, prob_distractor

    def get_img_sent_match_score(self, test_sentence, img_path):
        with open(img_path, "rb") as fp:
            binary = fp.read()
        image_bytes = io.BytesIO(binary)
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        image_tensor = torch.cat([tr(image) for tr in self.transforms]).unsqueeze(0)

        encoding = self.tokenizer(test_sentence, padding="max_length", truncation=True,
                                  max_length=self.config["max_text_len"], return_special_tokens_mask=True)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.int).unsqueeze(0)
        text_masks = torch.tensor(encoding["attention_mask"], dtype=torch.int).unsqueeze(0)

        # decoded_sequence = self.tokenizer.decode(encoding["input_ids"])
        # print(decoded_sequence)

        with torch.no_grad():
            text_embeds = self.model.text_embeddings(input_ids)
            (image_embeds, image_masks, patch_index, image_labels) = self.model.transformer.visual_embed(image_tensor,
                                                                                max_image_len=self.config["max_image_len"], mask_it=False)
            text_embeds, image_embeds = (text_embeds + self.model.token_type_embeddings(torch.zeros_like(text_masks)),
                image_embeds + self.model.token_type_embeddings(torch.full_like(image_masks, self.image_token_type_idx)),
            )

            x = torch.cat([text_embeds, image_embeds], dim=1)
            for i, blk in enumerate(self.model.transformer.blocks):
                x, _ = blk(x, mask=torch.cat([text_masks, image_masks], dim=1))
            outputs = self.model.transformer.norm(x)
            pooled = self.model.pooler(outputs)

        image_text_matching_score = self.model.itm_score(pooled)

        # Apply softmax
        softmaxed = F.softmax(image_text_matching_score[0], dim=0)
        # return the probability for image-sentence match:
        return softmaxed[1].item()


class OscarModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file):
        from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
        from transformers.pytorch_transformers import BertTokenizer, BertConfig

        super(OscarModelRepresentation, self).__init__(device)
        checkpoint = os.path.join("models/Oscar/checkpoint-29-132780")

        self.tokenizer = BertTokenizer.from_pretrained(checkpoint, local_files_only=True, do_lower_case=True)
        self.config = BertConfig.from_pretrained(checkpoint)
        self.config.img_layer_norm_eps = 1e-12
        self.config.use_img_layernorm = 0
        self.config.img_feature_dim = 2054
        self.config.img_feature_type = 'faster_r-cnn'
        self.config.attention_probs_dropout_prob = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.num_contrast_classes = 2
        self.max_img_seq_length = 50
        self.max_seq_length = 70
        self.model = ImageBertForSequenceClassification.from_pretrained(checkpoint, from_tf=False, config=self.config)
        self.initialisation(self.model)
        self.visual_feat_dict = load_visual_feats(visual_feats_file)

    def transform_visual_feats(self, visual_feats):
        boxes = visual_feats['boxes']
        features = visual_feats['features']
        image_height, image_width = visual_feats['img_h'], visual_feats['img_w']
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate(
             (scaled_x,
              scaled_y,
              scaled_x + scaled_width,
              scaled_y + scaled_height,
              scaled_width,
              scaled_height),
             axis=1)
        full_features = np.concatenate((features, spatial_features), axis=1)
        full_features = torch.from_numpy(full_features)
        return full_features

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, visual_features_path)
        return prob_target, prob_distractor

    def get_img_sent_match_score(self, test_sentence, visual_features_path):
        sentence_a_tokens = self.tokenizer.tokenize(test_sentence)

        visual_features_key = os.path.basename(visual_features_path)
        visual_feat_raw = self.visual_feat_dict[visual_features_key]

        img_labels = visual_feat_raw["object_names"]

        if len(sentence_a_tokens) + len(img_labels) > self.max_seq_length - 3:
            img_labels = img_labels[:(self.max_seq_length - len(sentence_a_tokens) - 3)]

        tokens = [self.tokenizer.cls_token] + sentence_a_tokens + [self.tokenizer.sep_token] + img_labels + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a_tokens) + 2 ) + [1] * (len(img_labels) + 1)
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        img_feat = self.transform_visual_feats(visual_feat_raw)

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            input_mask = input_mask + [1] * img_feat.shape[0]
        else:
            input_mask = input_mask + [1] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            input_mask = input_mask + ([0] * padding_matrix.shape[0])

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        img_feat = img_feat.unsqueeze(0).to(self.device)

        # decoded_sequence = self.tokenizer.decode(input_ids[0].numpy())
        # print(decoded_sequence)

        with torch.no_grad():
            logits = self.model(input_ids, token_type_ids=segment_ids,
                                      attention_mask=input_mask, img_feats=img_feat)

        softmaxed = F.softmax(logits[0][0], dim=0)
        # original caption: 1; polluted caption: 0
        return softmaxed[1].item()


class VINVLModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file):
        from transformers.pytorch_transformers import BertTokenizer, BertConfig
        from oscar.modeling.modeling_bert import BertImgForPreTraining
        from oscar.utils.tsv_file import TSVFile

        super(VINVLModelRepresentation, self).__init__(device)
        checkpoint = "models/VINVL/"
        self.config = BertConfig.from_pretrained(checkpoint)
        self.config.img_layer_norm_eps = 1e-12
        self.config.use_img_layernorm = 1
        self.config.img_feature_dim = 2054
        self.config.img_feature_type = 'faster_r-cnn'
        self.config.hidden_dropout_prob = 0.1
        self.config.num_contrast_classes = 3

        self.max_img_seq_length = 50
        self.max_seq_length = 35

        model_path = os.path.join(checkpoint, 'pytorch_model.bin')
        self.model = BertImgForPreTraining.from_pretrained(
            model_path,
            from_tf=False,
            config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)

        img_box_label_idx_to_label_path = os.path.join(checkpoint, 'VG-SGG-dicts-vgoi6-clipped.json')
        self.img_box_label_idx_to_label = json.load(open(img_box_label_idx_to_label_path))

        self.initialisation(self.model)
        print('Start loading img features')
        self.img_feature_file = torch.load(os.path.join(visual_feats_file, 'predictions.pth'))
        tsv_idx = pd.read_csv(os.path.join(visual_feats_file, "train.tsv"), delimiter="\t", header=None)
        self.hw = TSVFile(os.path.join(visual_feats_file, "train.hw.tsv"))
        self.img_idx_to_tsv_idx = {}
        for i, row in tsv_idx.iterrows():
            self.img_idx_to_tsv_idx[row[0]] = i
        print('End loading')

    def get_img_feat(self, img_id):
        idx = self.img_idx_to_tsv_idx[img_id]
        arr = self.img_feature_file[idx]
        boxes = arr.bbox
        num_boxes = len(boxes)
        feat = arr.get_field('box_features').numpy()
        feat = feat.reshape((num_boxes, 2048))
        info = json.loads(self.hw.seek(idx)[1])[0]
        image_height, image_width = info["height"], info["width"]
        box_width = boxes[:, 2] - boxes[:, 0]
        box_height = boxes[:, 3] - boxes[:, 1]
        scaled_width = box_width / image_width
        scaled_height = box_height / image_height
        scaled_x = boxes[:, 0] / image_width
        scaled_y = boxes[:, 1] / image_height
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]
        spatial_features = np.concatenate(
            (scaled_x,
             scaled_y,
             scaled_x + scaled_width,
             scaled_y + scaled_height,
             scaled_width,
             scaled_height),
            axis=1)
        full_features = np.concatenate((feat, spatial_features), axis=1)
        full_features = torch.from_numpy(full_features)
        return full_features

    def get_img_labels(self, img_id):
        idx = self.img_idx_to_tsv_idx[img_id]
        arr = self.img_feature_file[idx]
        labels = arr.extra_fields["labels"]
        labels_decoded = [self.img_box_label_idx_to_label["idx_to_label"][str(idx.item())] for idx in labels]
        return labels_decoded

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, visual_features_path)
        return prob_target, prob_distractor

    def get_img_sent_match_score(self, test_sentence, visual_features_path):
        sentence_a_tokens = self.tokenizer.tokenize(test_sentence)

        img_id = os.path.basename(visual_features_path).split('.')[0]
        img_labels = self.get_img_labels(img_id)

        if len(sentence_a_tokens) + len(img_labels) > self.max_seq_length - 3:
            img_labels = img_labels[:(self.max_seq_length - len(sentence_a_tokens) - 3)]

        tokens = [self.tokenizer.cls_token] + sentence_a_tokens + [self.tokenizer.sep_token] + img_labels + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a_tokens) + 2 ) + [1] * (len(img_labels) + 1)
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        img_feat = self.get_img_feat(img_id)

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            input_mask = input_mask + [1] * img_feat.shape[0]
        else:
            input_mask = input_mask + [1] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            input_mask = input_mask + ([0] * padding_matrix.shape[0])

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        img_feat = img_feat.unsqueeze(0).to(self.device)

        # decoded_sequence = self.tokenizer.decode(input_ids[0].numpy())
        # print(decoded_sequence)

        with torch.no_grad():
            outputs = self.model.bert(input_ids, token_type_ids=segment_ids,
                                      attention_mask=input_mask, img_feats=img_feat)
            sequence_output, pooled_output = outputs[:2]
            _, seq_relationship_score = self.model.cls(sequence_output, pooled_output)

        softmaxed = F.softmax(seq_relationship_score[0], dim=0)
        # 3-way contrastive loss: If triplet is matched, output should be 0
        # triplet is matched: 0; contains a polluted caption: 1; or contains a polluted answer: 2
        return softmaxed[0].item()


class VILBERTModelRepresentation(ModelRepresentation):
    def __init__(self, device, visual_feats_file):
        from transformers import BertTokenizer
        from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig

        super(VILBERTModelRepresentation, self).__init__(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        config_path = "src/vilbert-multi-task/config/bert_base_6layer_6conect.json"
        config = BertConfig.from_json_file(config_path)

        checkpoint = "models/VILBERT/pretrained_model.bin"
        self.model = BertForMultiModalPreTraining.from_pretrained(
            checkpoint, config=config
        )

        self.visual_feat_dict = load_visual_feats(visual_feats_file)

        self.initialisation(self.model)

    def get_img_sent_match_scores(self, target_sentence, distractor_sentence, visual_features_path):
        prob_target = self.get_img_sent_match_score(target_sentence, visual_features_path)
        prob_distractor = self.get_img_sent_match_score(distractor_sentence, visual_features_path)
        return prob_target.item(), prob_distractor.item()

    def get_img_sent_match_score(self, test_sentence, visual_features_path):
        visual_features_key = os.path.basename(visual_features_path)
        visual_feat_raw = self.visual_feat_dict[visual_features_key]
        input_ids, input_mask, segment_ids, img_feat, img_pos_feat, image_masks = transform_sample(self.tokenizer,
                                                                                  test_sentence,
                                                                                  visual_feat_raw,
                                                                                  num_locs=5,
                                                                                  max_seq_length=36,
                                                                                  add_global_imgfeat="first")

        # decoded_sequence = self.tokenizer.decode(input_ids[0])
        # print(decoded_sequence)

        with torch.no_grad():
            outputs = self.model(input_ids, img_feat, img_pos_feat, token_type_ids=segment_ids, attention_mask=input_mask, image_attention_mask=image_masks)
            _, _, seq_relationship_score, _ = outputs

        softmaxed = F.softmax(seq_relationship_score[0], dim=0)
        return softmaxed[0].data
