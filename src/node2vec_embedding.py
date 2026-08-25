# =============================================================
# src/node2vec_embedding.py
# Generates Node2Vec embeddings from WGCNA networks and
# concatenates UT + condition embeddings into 512-dim features.
# Estimated runtime: 3–7 hours per condition.
# =============================================================

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import multiprocessing
from config import (NETWORKS_PATH, FEATURES_PATH, CONDITIONS,
                    NODE2VEC_DIMENSIONS, NODE2VEC_WALK_LENGTH,
                    NODE2VEC_NUM_WALKS, NODE2VEC_WINDOW_SIZE,
                    NODE2VEC_P, NODE2VEC_Q, NODE2VEC_EPOCHS,
                    RANDOM_SEED)

np.random.seed(RANDOM_SEED)
WORKERS = min(8, multiprocessing.cpu_count())


def load_network(network_path):
    """Load edge list and build weighted NetworkX graph."""
    edge_df = pd.read_csv(network_path)
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    return G


def weighted_random_walk(G, start_node, walk_length, p, q):
    """Biased weighted random walk for Node2Vec."""
    walk = [start_node]
    for _ in range(walk_length - 1):
        cur       = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            weights = [G[cur][nbr]['weight'] for nbr in neighbors]
        else:
            prev    = walk[-2]
            weights = []
            for nbr in neighbors:
                w = G[cur][nbr]['weight']
                if nbr == prev:
                    weights.append(w / p)
                elif G.has_edge(nbr, prev):
                    weights.append(w)
                else:
                    weights.append(w / q)
        weights = np.array(weights, dtype=np.float32)
        total   = weights.sum()
        if total == 0:
            break
        probs   = weights / total
        walk.append(np.random.choice(neighbors, p=probs))
    return walk


def generate_walks(G, num_walks, walk_length, p, q):
    """Generate random walks for all nodes."""
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(weighted_random_walk(G, node, walk_length, p, q))
    return [[str(n) for n in walk] for walk in walks]


def train_embeddings(walks, dimensions, window_size, epochs):
    """Train Word2Vec skip-gram on random walks."""
    model = Word2Vec(
        sentences   = walks,
        vector_size = dimensions,
        window      = window_size,
        min_count   = 0,
        sg          = 1,
        workers     = WORKERS,
        epochs      = epochs,
    )
    return model


def get_embedding_df(model, nodes):
    """Extract L2-normalized embeddings as DataFrame."""
    vectors = np.array([model.wv[str(n)] for n in nodes])
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms
    return pd.DataFrame(vectors, index=nodes)


def run(networks_path=NETWORKS_PATH, features_path=FEATURES_PATH):
    print("=" * 55)
    print("  Step 3: Node2Vec Embedding")
    print("=" * 55)

    # Embed UT network
    print("\nEmbedding UT network...")
    G_ut    = load_network(f"{networks_path}/UT_WGCNA_weighted_t04.csv")
    walks   = generate_walks(G_ut, NODE2VEC_NUM_WALKS, NODE2VEC_WALK_LENGTH,
                              NODE2VEC_P, NODE2VEC_Q)
    model   = train_embeddings(walks, NODE2VEC_DIMENSIONS,
                                NODE2VEC_WINDOW_SIZE, NODE2VEC_EPOCHS)
    emb_ut  = get_embedding_df(model, list(G_ut.nodes()))
    print(f"  UT embedding: {emb_ut.shape}")

    # Embed each condition and concatenate with UT
    for cond_name in CONDITIONS:
        print(f"\nEmbedding {cond_name} network...")
        G_cond   = load_network(f"{networks_path}/{cond_name}_WGCNA_weighted_t04.csv")
        walks    = generate_walks(G_cond, NODE2VEC_NUM_WALKS, NODE2VEC_WALK_LENGTH,
                                  NODE2VEC_P, NODE2VEC_Q)
        model    = train_embeddings(walks, NODE2VEC_DIMENSIONS,
                                    NODE2VEC_WINDOW_SIZE, NODE2VEC_EPOCHS)
        emb_cond = get_embedding_df(model, list(G_cond.nodes()))

        # Concatenate UT + condition embeddings
        common = emb_ut.index.intersection(emb_cond.index)
        concat = pd.concat([emb_ut.loc[common], emb_cond.loc[common]], axis=1)
        concat.columns = range(concat.shape[1])

        out_path = f"{features_path}/Embeddings512_UT_{cond_name}_WGCNA_concat.csv"
        concat.to_csv(out_path)
        print(f"  {cond_name} embedding: {concat.shape} → {out_path}")

    print("\n✅ All embeddings saved.")


if __name__ == "__main__":
    run()
