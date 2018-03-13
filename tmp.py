import pickle

distances = {'euclidean': None,
             'cosine': None,
             # 'minkowski': None,
             #'chebyshev': None
             }

for key in distances.keys():
    with open('/home/beast/cocktails/pickle_objects/{}'.format(key), 'rb') as f:
        distances[key] = pickle.load(f).head(10)

print(distances)
with open('data_encoder', 'wb') as f:
    pickle.dump(distances, f)
