### LIBRARY

import tensorflow as tf
import evaluate
import seqeval


from transformers import create_optimizer
from transformers import TFAutoModelForTokenClassification
from transformers import RobertaTokenizerFast, DataCollatorForTokenClassification

print('     Library loaded')

PATH = {'train': 'Dataset/Train',
        'valid': 'Dataset/Valid'}

BATCH_SIZE = 16
NUM_EPOCH = 2
model_id = 'roberta-base'





tf_train_dataset = tf.data.experimental.load(
    PATH['train'], element_spec=None, compression=None, reader_func=None
)

tf_val_dataset = tf.data.experimental.load(
    PATH['valid'], element_spec=None, compression=None, reader_func=None
)

print('     Dataset loaded')
num_epoches = 2
batches_per_epoch = BATCH_SIZE*len(tf_train_dataset) // BATCH_SIZE
total_train_steps = int(num_epoches*batches_per_epoch)

optimizer, schedule = create_optimizer(init_lr = 2e-5, num_warmup_steps = 0, num_train_steps = total_train_steps)

model = TFAutoModelForTokenClassification.from_pretrained(model_id, num_labels = 9)

model.compile(optimizer = optimizer,
              metrics = ['accuracy'])

print('         ## Starting Training')

history = model.fit(tf_train_dataset,
                    validation_data = tf_val_dataset,
                    epochs = 2)


metric = evaluate.load('seqeval')
int_to_label = {0: 'O', 1: 'B-PER', 2 : 'I-PER', 3: 'B-ORG', 4 : 'I-ORG', 5 : 'B-LOC', 6: 'I-LOC', 7 : 'B-MISC', 8 : 'I-MISC'}

all_predictions = []
all_labels = []

for batch in tf_val_dataset:

  logits = model.predict(batch)['logits']
  labels = batch['labels'].numpy()

  batch_pred = []
  batch_labels = []

  predictions = tf.argmax(logits, axis = -1).numpy()

  for prediction, label in zip(predictions, labels):
    for predicted_idx, label_idx in zip(prediction, label):
      if label_idx == -100:
        continue

      batch_pred.append(int_to_label[predicted_idx])
      batch_labels.append(int_to_label[label_idx])

  all_predictions.append(batch_pred)
  all_labels.append(batch_labels)


seqeval = evaluate.load('seqeval')
predictions = all_predictions
references = all_labels
results = seqeval.compute(predictions=predictions, references=references)

print(results)