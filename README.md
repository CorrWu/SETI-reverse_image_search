# Reverse Image Search

## Summary 
My goal is to do reverse image search on a target interference patter and find the other interference similar to it. I tried using ResNet50 and a model developed by Peter Ma. In the end, ResNet50 worked out better for the purpose of reverse image search. 

## Exploration
I found some interference such as the following from `mnt_blpd7/datax/dl/GBT_57436_51432_HIP77257_fine.h5`.
![Screen Shot 2022-11-30 at 23 55 48](https://user-images.githubusercontent.com/67254464/204997060-57c5d441-75f9-4649-9762-d3908bc51eb4.png)
![Screen Shot 2022-11-30 at 23 57 23](https://user-images.githubusercontent.com/67254464/204997421-4bf38332-256d-4ed9-b19b-c9bb0a2d87db.png)
I chose a much smaller portion of the data as the target interference to improve performance.


## Data Preprocessing
```
def preprocess_input(data):
    log_input = np.log(data)
    scale_input = (log_input - log_input.min()) / log_input.max()
    return scale_input
```

## Feature Extraction
```
def extract_features(input_arr, model):
    input_shape = (6, 256, 1)
    normalized_arr = local_contrast_normalization(input_arr)
    features = model.predict(normalized_arr)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features
```

## Generating Input Data
```
start = 1520
stop = 1525
interval = 1
step = 0.5
data_list = []
data = blimpy.Waterfall(url, load_data=True, f_start=start, f_stop=stop)
for i in np.arange(start, stop, step):
    fstart, fstop = round(i, 3), i + interval
    _, sub_data = data.grab_data(f_start=fstart, f_stop=fstop)
    resized_data = resize(sub_data, (1, 6, 256, 1))
    data_list.append(resized_data)
```

## Generating Features
```
feature_list = []
for i in range(len(data_list)):
    data = data_list[i]
    feature_list.append(extract_features(data, vae.encoder))
```

## Finding the Nearest Neighbor
```
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(feature_list)
```
