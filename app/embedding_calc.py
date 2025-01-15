from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP


# model modified to output 256 dim embeddings
transformer_model = SentenceTransformer("/app/stella_en_400M_v5", trust_remote_code=True)
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)



def calc_reduced_embeddings(texts: list[str]):
    transformer_embeddings = transformer_model.encode(texts)
    #umap_embeddings = umap_model.fit_transform(transformer_embeddings)
    #return umap_embeddings
    return transformer_embeddings
    