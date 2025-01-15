A server for computing embeddings of sentences. client / server communication using protobuf.

requirements:

* CUDA GPU and driver installed on host docker machine for CUDA docker support

steps to run:
---
1) cd into the project directory.
1) run `git clone --depth 1 https://huggingface.co/dunzhang/stella_en_400M_v5`
1) for generating embeddings of 256 instead of 1024:
   * in the stella_en_400M_v5 dir, modify `modules.json` and replace `2_Dense_1024` with `2_Dense_256` 
1) run `docker build -t sentence_embeddings_server .`.\
1) run ` docker run -it --rm --gpus=all --name=sentence_embeddings_server -v .../sentence-embedding-server/stella_en_400M_v5:/app/stella_en_400M_v5 -p 50051:50051 sentence_embeddings_server`