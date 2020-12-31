# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:24:12 2020

@author: swbatta
"""
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import torch
import random
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, BertConfig, BertModel

app = Flask(__name__)


label_dict = {'Anxiety': 0, 'Depression': 1, 'Relationship Stress': 2, 'Gender issues': 3}

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",

                                                      num_labels=len(label_dict),

                                                      output_attentions=False,

                                                      output_hidden_states=False)

model.load_state_dict(torch.load('.\\epoch_data1\\finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))

if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device=torch.device('cpu')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    
    #For rendering results on HTML GUI
    
    int_features = [x for x in request.form.values()]
    print(int_features)
    
    test_data = pd.DataFrame(int_features,columns = ['Query'])

    test_data.label = ""
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    encoded_data_test = tokenizer.batch_encode_plus(
    test_data.Query.values.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    truncation=True,
    max_length=256, 
    return_tensors='pt')


    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    
    dataset_test = TensorDataset(input_ids_test, attention_masks_test)
    
    batch_size = 3
    
    dataloader_test = DataLoader(dataset_test, 
                                       sampler=SequentialSampler(dataset_test), 
                                       batch_size=batch_size)
    
    predictions = []
    
    for d in dataloader_test:
        d=tuple(t.to(device) for t in d)
        
        b_input_ids, b_input_mask = d
        
        with torch.no_grad():
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
        logits = outputs[0]
        
        logits = logits.detach().cpu().numpy()
        
        predictions.append(logits)
    
    
    flat_predictions = [item for sublist in predictions for item in sublist]
    
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    for i in label_dict:
        if (label_dict[i] == flat_predictions):
            print(i)
            output = i
    
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = int_features

    '''
    Importing data containing links to articles
    '''
    articles_df = pd.read_csv(r".\\Rawdata\\better_help_articles.csv")
    articles = pd.Series(articles_df[articles_df['Topic']==output]['Articles'])
    articles_list_values = articles.values[0][1:-1].split(",")

    random_article = random.choice(articles_list_values).strip()[1:-1]

    return render_template('result.html', prediction_text='Your concern might be related to {}'.format(output), articles=random_article)
'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
'''

if __name__ == "__main__":
    app.run(debug=True)