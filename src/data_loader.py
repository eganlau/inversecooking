# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image, ImageFile
from build_vocab import Vocabulary
import random
import json
import lmdb
import mysql.connector

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_file_path = "/media/eganlau/Meal Pictures/Images"
appearance_threshold = 0.5

def make_dataset():
    db_config = {
                'user': 'analyze',
                'password': 'Fittime1991,',
                'host': 'localhost',
                'port': '3306'
                }

    idx2word = {}
    idx2word[0] = '<end>'
    word2idx = {}
    word2idx['<end>'] = 0
    i = 1
    try:
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor(dictionary=True)

        sql = '''
                SELECT * FROM app.ingredients_appearance order by cum_percent;
              '''
        cur.execute(sql)

        for row in cur:
            if row["cum_percent"] <= appearance_threshold:
                idx2word[i] = [row["ingredients_name"]]
                word2idx[row["ingredients_name"]] = i
                i += 1
        idx2word[i] = ['<pad>']
        word2idx['<pad>'] = i
        i += 1
        # print(idx2word)
        # print(word2idx)

        vocab_ingrs = Vocabulary()
        vocab_ingrs.idx2word = idx2word
        vocab_ingrs.word2idx = word2idx
        vocab_ingrs.i = i
        vocab_toks = vocab_ingrs
        dataset = []
        # dataset['train'] = []
        # dataset['val'] = []
        # dataset['test'] = []
        
        sql = f'''
            select * from hotcamp.tb_checkin_detail 
            where exists (select ingredients_appearance.ingredients_id
                from app.ingredients_appearance 
                where ingredients_appearance.cum_percent <= {appearance_threshold} 
                and ingredients_appearance.ingredients_id = tb_checkin_detail.ingredients_id)
            order by tb_checkin_detail.apply_id, tb_checkin_detail.date, tb_checkin_detail.type
            LIMIT 1000
            '''
        cur.execute(sql)

        records = {}
        for row in cur:
            date_stamp = row['date'][0:4]+row['date'][5:7]+row['date'][8:10]
            record_id = str(row['apply_id'])+'_'+date_stamp+"_"+row['type']
            if record_id not in records:
                records[record_id] = {}
                records[record_id]["id"] = record_id
                records[record_id]["instructions"] = ["1","2","3"]
                records[record_id]["tokenized"] = ["1","2","3"]
                records[record_id]["ingredients"] = []
                records[record_id]["images"] = []
                records[record_id]["title"] = ["1","2","3"]
                
            file_path = date_stamp+'/hotcamp_'+record_id+".jpg"
            if os.path.isfile(os.path.join(image_file_path, file_path)):
                records[record_id]["ingredients"].append(row["name"])
                records[record_id]["images"].append(file_path)
                # print(f"{file_path} exists")
            # else:
            #     print(f"{file_path} does not exists")

        # split_chance = np.random.rand(len(records))
        for record in records.values():
            dataset.append(record)
        #     chance = np.random.random_sample()
        #     if  chance <= 0.8:
        #         dataset['train'].append(record)
        #     elif chance > 0.8 and chance <= 0.9:
        #         dataset['val'].append(record)
        #     else:
        #         dataset['test'].append(record)

        # print("vocab length: ",len(vocab_ingrs))
        cur.close()
        conn.commit()
        conn.close()
    except mysql.connector.Error as e:
        print("Mysql Error %d: %s" % (e.args[0], e.args[1]))
    return vocab_ingrs, vocab_toks, dataset


class Recipe1MDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir, split, maxseqlen, maxnuminstrs, maxnumlabels, maxnumims,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff=''):

        # self.ingrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_ingrs.pkl'), 'rb'))
        # self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_toks.pkl'), 'rb'))
        # self.dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_'+split+'.pkl'), 'rb'))

        self.ingrs_vocab, self.instrs_vocab, self.dataset = make_dataset()
        self.label2word = self.get_ingrs_vocab()
        # print("len(self.instrs_vocab): ", len(self.instrs_vocab))
        self.use_lmdb = use_lmdb
        # if use_lmdb:
        #     self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + split), max_readers=1, readonly=True,
        #                                 lock=False, readahead=False, meminit=False)

        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            # print(entry)
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.root = os.path.join(data_dir, 'images', split)
        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        img_id = sample['id']
        captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]

        idx = index

        labels = self.dataset[self.ids[idx]]['ingredients']
        title = sample['title']

        tokens = []
        tokens.extend(title)
        # add fake token to separate title from recipe
        tokens.append('<eoi>')
        for c in captions:
            tokens.extend(c)
            tokens.append('<eoi>')

        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label)
            if label_idx not in ilabels_gt:
                ilabels_gt[pos] = label_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingrs_gt = torch.from_numpy(ilabels_gt).long()

        if len(paths) == 0:
            path = None
            image_input = torch.zeros((3, 224, 224))
        else:
            if self.split == 'train':
                img_idx = np.random.randint(0, len(paths))
            else:
                img_idx = 0
            path = paths[img_idx]
            # if self.use_lmdb:
            #     try:
            #         with self.image_file.begin(write=False) as txn:
            #             image = txn.get(path.encode())
            #             image = np.fromstring(image, dtype=np.uint8)
            #             image = np.reshape(image, (256, 256, 3))
            #         image = Image.fromarray(image.astype('uint8'), 'RGB')
            #     except:
            #         print ("Image id not found in lmdb. Loading jpeg file...")
            #         image = Image.open(os.path.join(self.root, path[0], path[1],
            #                                         path[2], path[3], path)).convert('RGB')
            # else:
            #    image = Image.open(os.path.join(self.root, path[0], path[1], path[2], path[3], path)).convert('RGB')
            # if os.path.isfile(os.path.join(image_file_path, path)):
            #     print("path exists: ",os.path.join(image_file_path, path))
            image = Image.open(os.path.join(image_file_path, path)).convert('RGB')
                
            if self.transform is not None:
                image = self.transform(image)
            image_input = image

        # Convert caption (string) to word ids.
        caption = []

        caption = self.caption_to_idxs(tokens, caption)
        caption.append(self.instrs_vocab('<end>'))

        caption = caption[0:self.maxseqlen]
        target = torch.Tensor(caption)

        return image_input, target, ingrs_gt, img_id, path, self.instrs_vocab('<pad>')

    def __len__(self):
        return len(self.ids)

    def caption_to_idxs(self, tokens, caption):

        caption.append(self.instrs_vocab('<start>'))
        for token in tokens:
            caption.append(self.instrs_vocab(token))
        return caption


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return image_input, targets, ingrs_gt, img_id, path


def get_loader(data_dir, aux_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels, maxnumims, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff=''):

    dataset = Recipe1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset

if __name__ == '__main__':
    make_dataset()