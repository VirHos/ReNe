docker build -t rene .
docker run -e PYTHONPATH="/rene/src" \
    -v "$(pwd)/src:/rene/src" -v "$(pwd)/scripts:/rene/scripts" -v "$(pwd)/data:/rene/data" \
    -w /rene rene \
    python3 scripts/build_user_history.py && python3 scripts/build_cache.py
#docker rm -f rene_cache