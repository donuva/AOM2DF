import torch
import numpy as np
import json
import csv
import os
import json
from itertools import chain
import torch.utils.data as data
import pickle as pkl
from transformers import AutoTokenizer
from sklearn.decomposition import PCA

def get_img_region_box(img_region_dir,id):
    region_feat = np.load(
        os.path.join(img_region_dir + '/_att',
                     id[:-4] + '.npz'))['feat']
    box = np.load(
        os.path.join(img_region_dir + '/_box', id[:-4] + '.npy'))

    return region_feat, box


def get_aesc_spans( dic):
    aesc_spans = []
    for x in dic:
        aesc_spans.append((x['from'], x['to'], x['polarity']))
    return aesc_spans


def get_gt_aspect_senti( dic):
    gt = []
    for x in dic:
        gt.append((' '.join(x['term']), x['polarity']))
    return gt

def getData(infos,split):
    infos = json.load(open(infos, 'r'))

    if split == 'train':
        data_set = json.load(
            open(infos['data_dir'] + '/train.json', 'r'))
        img_region_dir = infos['img_region_dir']
    elif split == 'dev':
        data_set = json.load(
            open(infos['data_dir'] + '/dev.json', 'r'))
        img_region_dir = infos['img_region_dir']
    elif split == 'test':
        data_set = json.load(
            open(infos['data_dir'] + '/test.json', 'r'))
        img_region_dir = infos['img_region_dir']
    else:
        raise RuntimeError("split type is not exist!!!")

    total_batch = []
    for data in data_set:
        output = {}
        img_id = data['image_id']
        region_feat, box = get_img_region_box(img_region_dir,img_id)
        # img_feature = np.concatenate([region_feat, box], axis=1)  #check 维度
        img_feature = region_feat
        #u_pos = data['u_pos']
        #output['u_pos'] = u_pos
        output['img_feat'] = img_feature

        output['sentence'] = ' '.join(data['words'])

        aesc_spans = get_aesc_spans(data['aspects'])
        output['aesc_spans'] = aesc_spans
        output['image_id'] = img_id
        gt = get_gt_aspect_senti(data['aspects'])
        output['gt'] = gt
        total_batch.append(output)
    return total_batch


class Preprocess:
    def __init__(self,args,output):
        self.args = args
        self.input_ids = output['input_ids']
        self.attention_mask = output['attention_mask']
        self.image_id = output['image_id']
        self.image_features = output['image_features']
        self.image_features = np.array(self.image_features,dtype=object)
        #m2df
        self.sentiment_value = output['sentiment_value']
        #self.sentiment_value = np.array(self.sentiment_value,dtype=object)
        self.noun_mask = output['noun_mask']
        #self.noun_mask = np.array(self.noun_mask,dtype=object)
        self.dependency_matrix = output['dependency_matrix']
        #
        self.AESC = output['AESC']
        self.span_labels = self.AESC['labels']
        #self.span_labels = np.array(self.span_labels,dtype=object)
        self.span_masks = self.AESC['masks']
        self.gt_spans = self.AESC['spans']
        self.gt_spans = np.array(self.gt_spans, dtype=object)
        self.task = output['task']


        self.image_text_similarity_path = "./Data_New/m2df/m2df_whole2015.json"
        self.image_text_region_similarity_path = "./Data_New/m2df/m2df_region2015.json"

        self.sequence_id = np.arange(len(self.input_ids))

        self.image_text_similarity = self.calculate_image_text_similarity()
        self.image_text_region_similarity = self.calculate_image_text_region_similarity()

        #similarity
        self.input_ids_by_similarity, self.attention_masks_by_similarity, \
        self.sentiment_value_by_similarity, self.noun_mask_by_similarity, self.dependency_matrix_by_similarity,\
        self.image_ids_by_similarity, self.image_feats_by_similarity, \
        self.span_labels_by_similarity, self.span_masks_by_similarity, \
        self.gt_spans_by_similarity = self.sort_by_modal_similarity_difficulty()

        #region similarity
        self.input_ids_by_region_similarity, self.attention_masks_by_region_similarity, \
        self.sentiment_value_by_region_similarity, self.noun_mask_by_region_similarity, self.dependency_matrix_by_region_similarity,\
        self.image_ids_by_region_similarity, self.image_feats_by_region_similarity, \
        self.span_labels_by_region_similarity, self.span_masks_by_region_similarity, \
        self.gt_spans_by_region_similarity = self.sort_by_modal_region_similarity_difficulty()

    def calculate_image_text_similarity(self):
        with open(self.image_text_similarity_path,'r',encoding='utf-8')as f:
            similarity_dict = json.load(f)
        similarity_list = []
        for image_id in self.image_id:
            similarity_list.append(abs(similarity_dict[image_id]))
        return similarity_list

    def calculate_image_text_region_similarity(self):
        with open(self.image_text_region_similarity_path,'r',encoding='utf-8')as f:
            similarity_dict = json.load(f)
        similarity_list = []
        for image_id in self.image_id:
            similarity_list.append(abs(similarity_dict[image_id]))
        return similarity_list

    

    def sort_by_modal_similarity_difficulty(self):
        sort_index = np.argsort(self.image_text_similarity)


        image_ids = np.array(self.image_id, dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]
        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]
        #m2df
        sentiment_value = self.sentiment_value[sort_index]
        noun_mask = self.noun_mask[sort_index]
        dependency_matrix = self.dependency_matrix[sort_index]
        

        return input_ids, atttention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans,sentiment_value,noun_mask,dependency_matrix

    def sort_by_modal_region_similarity_difficulty(self):
        sort_index = np.argsort(self.image_text_region_similarity)


        image_ids = np.array(self.image_id, dtype=object)
        # 排序
        input_ids = self.input_ids[sort_index]
        atttention_masks = self.attention_mask[sort_index]
        image_ids = image_ids[sort_index]
        image_feats = self.image_features[sort_index]
        span_labels = self.span_labels[sort_index]
        span_masks = self.span_masks[sort_index]
        gt_spans = self.gt_spans[sort_index]
        #m2df
        sentiment_value = self.sentiment_value[sort_index]
        noun_mask = self.noun_mask[sort_index]
        dependency_matrix = self.dependency_matrix[sort_index]
        

        return input_ids, atttention_masks, image_ids, image_feats, span_labels, span_masks, gt_spans,sentiment_value,noun_mask,dependency_matrix

    def get_sample_batch_by_similarity(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int((lambda_init + (1 - lambda_init) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init**2 + (1 - lambda_init**2) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        current_index = min(len(self.input_ids_by_similarity)-1,current_index)
        batch = [self.input_ids_by_similarity[:current_index],
                 self.attention_masks_by_similarity[:current_index],
                 self.image_ids_by_similarity[:current_index],
                 self.image_feats_by_similarity[:current_index],
                 self.span_labels_by_similarity[:current_index],
                 self.span_masks_by_similarity[:current_index],
                 self.gt_spans_by_similarity[:current_index],
                 #m2df
                 self.sentiment_value_by_similarity[:current_index], self.noun_mask_by_similarity[:current_index], self.dependency_matrix_by_similarity[:current_index]
                 ]
        return batch

    def get_sample_batch_by_region_similarity(self,lambda_init,current_epoch,total_epoch):
        if self.args.curriculum_pace == 'linear':
            current_index = int((lambda_init + (1 - lambda_init) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        elif self.args.curriculum_pace == 'square':
            current_index = int((lambda_init**2 + (1 - lambda_init**2) * current_epoch/total_epoch) * len(self.input_ids_by_similarity))
        current_index = min(len(self.input_ids_by_region_similarity)-1,current_index)
        batch = [self.input_ids_by_region_similarity[:current_index],
                 self.attention_masks_by_region_similarity[:current_index],
                 self.image_ids_by_region_similarity[:current_index],
                 self.image_feats_by_region_similarity[:current_index],
                 self.span_labels_by_region_similarity[:current_index],
                 self.span_masks_by_region_similarity[:current_index],
                 self.gt_spans_by_region_similarity[:current_index],
                 #m2df
                 self.sentiment_value_by_region_similarity[:current_index], self.noun_mask_by_region_similarity[:current_index], self.dependency_matrix_by_region_similarity[:current_index]
                 ]
        return batch

class Dataset(data.Dataset):
    def __init__(self,args,input_ids,attention_masks,image_feats,span_labels,span_masks,sentiment_value,noun_mask,dependency_matrix):
        self.input_ids = torch.tensor(input_ids).to(args.device)
        self.attention_masks = torch.tensor(attention_masks).to(args.device)
        #print(image_feats)
        self.image_feats = image_feats
        self.span_labels = torch.tensor(span_labels).to(args.device)
        self.span_masks = torch.tensor(span_masks).to(args.device)
        self.data_ids = torch.arange(len(input_ids)).to(args.device)#len of input_ids
        self.sentiment_value = sentiment_value#torch.tensor(sentiment_value).to(args.device)
        self.noun_mask = noun_mask#torch.tensor(noun_mask).to(args.device)
        self.dependency_matrix = dependency_matrix#torch.tensor(dependency_matrix).to(args.device)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, index):
        # print(self.input_ids[index].shape)
        # print(self.attention_masks[index].shape)
        # print(self.image_feats[index].shape)
        # print(self.span_labels[index].shape)
        # print(self.span_masks[index].shape)
        # print(self.data_ids[index].shape)
        # print(self.sentiment_value[index].shape)
        # print(self.noun_mask[index].shape)
        # print(self.dependency_matrix[index].shape)
        return self.input_ids[index],self.attention_masks[index],self.image_feats[index],self.span_labels[index],\
               self.span_masks[index],self.data_ids[index],\
               self.sentiment_value[index], self.noun_mask[index], self.dependency_matrix[index]


