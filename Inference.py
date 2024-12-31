import tensorflow as tf
import pandas as pd
from transformers import TFAutoModelForTokenClassification, RobertaTokenizerFast


model_id = 'roberta-base'

ner_model = TFAutoModelForTokenClassification.from_pretrained('NER_Model', num_labels = 9)
tokenizer = RobertaTokenizerFast.from_pretrained(model_id, add_prefix_space=True)

ind_to_label={0:'O', 1:'B-PER',2:'I-PER',3:'B-ORG',4:'I-ORG',5:'B-LOC',6:'I-LOC',7:'B-MISC',8:'I-MISC'}
out_str=""
current_index=0

user_input = None


rez = []

while user_input != '/exit':

    user_input = input('Give example: ')
    out_str = ''

    inputs= tokenizer([user_input], padding=True,return_tensors="tf")
    logits = ner_model(**inputs).logits


    for i in range(1,len(inputs.tokens())-1):
        if tf.argmax(logits,axis= -1)[0][i]!=0:
            out_str+=" " + str(inputs.tokens()[i]) + "["+str(ind_to_label[tf.argmax(logits,axis= -1).numpy()[0][i]])+']'
        else:
            out_str+=" " + str (inputs.tokens()[i])

    
        
    rez.append([user_input, out_str.replace("Ġ","")])
    print(out_str.replace("Ġ",""))

pd.DataFrame(rez).iloc[:-1, :].to_csv('Result/test.csv')