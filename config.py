import transformers


MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL_BASE_PATH = 'bert-base-cased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = 'data/ner_dataset.csv'
LEARING_RATE = 3e-5

# The Bert implementation comes with a pretrained tokenizer and a definied vocabulary. We load the one related to the
# smallest pre-trained model bert-base-cased.
# We use the cased variate since it is well suited for NER.
TOKENZIER = transformers.BertTokenizer.from_pretrained(MODEL_BASE_PATH, do_lower_case=False)



