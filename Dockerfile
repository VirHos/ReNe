FROM tensorflow/tensorflow:1.15.5-py3
RUN pip3 install pyyaml==5.1.2 \
            streamlit==0.82.0 \
            faiss-cpu==1.7.1.post2 \
            flask==1.0.2 \
            wordpiece_tokenizer==1.1 