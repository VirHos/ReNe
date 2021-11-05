#docker build -t rene .
docker run -w /ReNe -v ${PWD}:/ReNe -p 5000:5000 -e "PYTHONPATH=/rene/src" rene python3 src/rest_wrapper.py