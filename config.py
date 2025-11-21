import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'model', 'passmodel.pkl'))
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', os.path.join(BASE_DIR, 'model', 'tfidfvectorizer.pkl'))
DATA_PATH = os.environ.get('DATA_PATH', os.path.join(BASE_DIR, 'data', 'custom_dataset.csv'))

SECRET_KEY = os.environ.get('SECRET_KEY', 'change-this-in-prod')
