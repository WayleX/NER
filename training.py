import random
import os
import spacy
from tqdm import tqdm 
from dataset import get_data
from spacy.tokens import DocBin
from spacy.util import filter_spans

def train():
    #read data and make dev and train sets in proporion 1 to 4
    all_data = get_data()
    dev_data =[]
    training_data = []
    random.shuffle(all_data)    
    for i in range(len(all_data)):
        if i%5 == 0:
            dev_data.append(all_data[i])
        else:
            training_data.append(all_data[i])
    #Create train dataset file using docbin
    nlp = spacy.load("en_core_web_lg")
    doc_bin = DocBin()
    for training_example  in tqdm(training_data): 
        text = training_example['text']
        labels = training_example['entities']
        doc = nlp.make_doc(text) 
        ents = []
        for start, end, label in labels:
            #Fix for including mount in dataset
            if 'mount' in text[start:end].lower():
                print(text[start:end], text[start:end].index(' '))
                start = text[start:end].index(' ')
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            #checking if word is correct
            if span is None:
                span = doc.char_span(start, end+1, label=label, alignment_mode="contract")
                if span is None:
                    #print("Skipping entity")
                    pass
                else:
                    ents.append(span)
            else:
                ents.append(span)
        i += 1
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents 
        doc_bin.add(doc)
    doc_bin.to_disk("training_data.spacy")
    #create dev dataset file
    doc_bin = DocBin()
    for training_example  in tqdm(dev_data): 
        text = training_example['text']
        labels = training_example['entities']
        doc = nlp.make_doc(text) 
        ents = []
        for start, end, label in labels:
            if 'mount' in text[start:end].lower():
                try:
                    start = text[start:end].index(' ')
                except:
                    continue
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            #checking if word is correct
            if span is None:
                span = doc.char_span(start, end+1, label=label, alignment_mode="contract")
                if span is None:
                    #print("Skipping entity")
                    pass
                else:
                    ents.append(span)
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents 
        doc_bin.add(doc)
    doc_bin.to_disk("dev_data.spacy")
    #run training from terminal
    os.system('python3 -m spacy init fill-config ./config/base_config.cfg ./config/config.cfg')
    os.system('python3 -m spacy train ./config/config.cfg --output ./ --paths.train ./data/training_data.spacy --paths.dev ./data/dev_data.spacy')