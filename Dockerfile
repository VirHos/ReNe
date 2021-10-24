FROM huggingface/transformers-cpu

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TRANSFORMERS_CACHE=/rene/data/hg_cache/

RUN pip3 install pyyaml==5.1.2 \
            streamlit==0.82.0 \
            faiss-cpu==1.7.1.post2 \
            flask==1.0.2 \
            tqdm \
            xlrd==1.1.0