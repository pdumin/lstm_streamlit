import streamlit as st
from utils.preprocessing import preprocess_single_string
import json 
from models.model import LSTMClassifier
import torch

with open('utils/vocab2int.json', 'r') as f:
    vocab_to_int = json.load(f)

@st.cache_resource
def load_model():
    clf = LSTMClassifier()
    clf.load_state_dict(torch.load('models/lstm_weights.pt'))
    clf.eval()
    return clf 

clf = load_model()


user_review = st.text_input(label='Input your review here')
classify = st.button('Classify!')

if user_review and classify:
    # st.write(vocab_to_int['movie'])
    preprocessed_review = preprocess_single_string(
        user_review, 
        seq_len=32, 
        vocab_to_int=vocab_to_int
    )
    # st.write(preprocessed_review.shape)
    with torch.inference_mode():
        out = clf(preprocessed_review.unsqueeze(0))
    st.write(f'Probability of positive sentiment: {out.item():.3f}')