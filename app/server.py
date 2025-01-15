from concurrent import futures
import zstandard as zstd
import logging
import json

import grpc
import embedding_pb2
import embedding_pb2_grpc

compressor = zstd.ZstdCompressor()
decompressor = zstd.ZstdDecompressor()

def np_ndarray_as_bytes(arr):
    import io
    import numpy as np

    memory_file = io.BytesIO()
    np.save(memory_file, arr)
    return memory_file.getvalue()


class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
        
    def CalculateEmbeddings(self, request, context):
        from embedding_calc import calc_reduced_embeddings
        
        logging.info("Received CalculateEmbeddings request")
        
        response = embedding_pb2.EmbeddingResponse()
        
        try:
            
            logging.info(f'Received request with {len(request.embeddingsListBinary)=}')
            
            texts_json_str = decompressor.decompress(request.embeddingsListBinary)
            texts_list = json.loads(texts_json_str)
            
            embeddings = calc_reduced_embeddings(texts_list)
            embeddings_bytes = np_ndarray_as_bytes(embeddings)
            embeddings_bytes_compressed = compressor.compress(embeddings_bytes)
            
            logging.info("Successfully processed CalculateEmbeddings request")
            
            response.embeddingsListBinary = embeddings_bytes_compressed
        except Exception as e:
            logging.error(f"Error processing CalculateEmbeddings request: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
        
        return response

    def Echo(self, request, context):
        return request

def serve():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting server")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started, listening on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
