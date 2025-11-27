from sentence_transformers import SentenceTransformer
import numpy as np


class BGE_M3_Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")


    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        print(f"Embedded {len(texts)} texts using BGE-M3 model.")
        print(f"First vector (truncated): {vectors[0][:5]}...")
        print("Vector shape:", np.array(vectors).shape)

        return np.array(vectors).tolist()


def get_embedder():
    return BGE_M3_Embedder()


# obj_bge_m3_embedder = BGE_M3_Embedder()

# v_en = obj_bge_m3_embedder.embed("I love cats!")
# v_bn = obj_bge_m3_embedder.embed("আমি বিড়ালকে ভালোবাসি!")
# v_es = obj_bge_m3_embedder.embed("¡Me encantan los gatos!")

# v_en = np.array(v_en[0])
# v_bn = np.array(v_bn[0])
# v_es = np.array(v_es[0])

# def cosine(a, b): return np.dot(a, b)

# print("EN–BN similarity:", cosine(v_en, v_bn))
# print("EN–ES similarity:", cosine(v_en, v_es))
# print("BN–ES similarity:", cosine(v_bn, v_es))