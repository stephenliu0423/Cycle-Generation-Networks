import numpy as np


def mmr(action_emb, movie_embs, K, can_set, l):
    s1 = []
    vector = []
    # action = action_emb[0]
    for _ in range(l):
        for i in range(K):
            action = action_emb[i]

            dis_scores = []
            for j in range(movie_embs.shape[0]):
                # dis_scores.append(mean_squared_error(action, movie_embs[j]))
                dis_scores.append(np.sum((action - movie_embs[j]) ** 2))
            dis_scores = np.array(dis_scores)

            max_num = np.max(dis_scores)

            ind = np.where(dis_scores == np.min(dis_scores))[0][0]
            dis_scores[ind] = max_num
            while ind not in can_set:
                ind = np.where(dis_scores == np.min(dis_scores))[0][0]
                dis_scores[ind] = max_num
            s1.append(ind)
            vector.append(movie_embs[ind])
            can_set.remove(ind)
    vector = np.array(vector)
    return s1, vector
