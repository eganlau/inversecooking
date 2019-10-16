import pickle
import os
import csv

data_dir = '../data'

ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
print(ingrs_vocab)
wtr = csv.writer(open (os.path.join(data_dir,'ingrs_vocab.csv'), 'w'), delimiter=',', lineterminator='\n')

wtr.writerow (["ingredients_name_english"])
# remove the <end>, <pad> first and last entries
for x in ingrs_vocab[1:len(ingrs_vocab)-1] : wtr.writerow ([x])