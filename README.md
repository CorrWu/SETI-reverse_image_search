# Reverse Image Search

## Summary 
My goal is to do reverse image search on a target interference pattern to find other interference patterns similar to it. 

## Exploration
I found some interference such as the following from `mnt_blpd7/datax/dl/GBT_57436_51432_HIP77257_fine.h5`.
![Screen Shot 2022-11-30 at 23 55 48](https://user-images.githubusercontent.com/67254464/204997060-57c5d441-75f9-4649-9762-d3908bc51eb4.png)  
![Screen Shot 2022-11-30 at 23 57 23](https://user-images.githubusercontent.com/67254464/204997421-4bf38332-256d-4ed9-b19b-c9bb0a2d87db.png)  
I chose a much smaller portion of the data as the target interference to improve performance.  
![Screen Shot 2022-12-04 at 16 53 03](https://user-images.githubusercontent.com/67254464/205526913-48a24cf8-357a-46e0-80f2-f36931abf7c3.png)
  

## Model
I tried using ResNet50 and a model developed by Peter Ma. In the end, ResNet50 with imagenet worked out better for the purpose of reverse image search. 
```
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

## Generating Input Data
The variable `interval` is the difference between the starting point (`f_start`) and the stopping point (`f_stop`) of each image. The variable `step` is the difference between the starting points (`f_start`) of each image. Both variables can be changed if needed.
I set the `interval` to a small number (256 * 2.79e-6) to improve the efficiency of the program. The variable `step` can be set to a smaller number to get more accurate result.  
![Screen Shot 2022-12-04 at 16 32 15](https://user-images.githubusercontent.com/67254464/205525951-5ca9a7eb-44fc-4c1e-ab40-371bea66059d.png)  
I used `skimage.transform.resize` to resize each interval of frequency to the shape of (1, 224, 224, 3), which is the shape ResNet50 uses. The resulting `data_list` is a list of arrays with the shape. Each array is perceived as an image when passed in the feature extraction function. They are compared with each other to find the nearest neighbor among them. 
```
start = 1530
stop = 1535
interval = 256 * 2.79e-6    # The difference between the starting point (`f_start`) and the stopping point (`f_stop`)
step = interval             # The difference between the starting points (`f_start`) of each interval
data_list = []
wf = blimpy.Waterfall(url, load_data=True, f_start=start, f_stop=stop)
for i in np.arange(start, stop, step):
    fstart, fstop = round(i, 3), i + interval
    _, sub_data = wf.grab_data(f_start=fstart, f_stop=fstop)
    resized_data = resize(sub_data, (1, 224, 224, 3))
    data_list.append(resized_data)
```

## Data Preprocessing
I used logarithm on the data and then scaled the data to numbers between 0 and 1. 
```
def preprocess_input(data):
    log_input = np.log(data)
    scale_input = (log_input - log_input.min()) / log_input.max()
    return scale_input
```

## Feature Extraction
This function preprocesses the input array using the `preprocess_input` function above. Then it generates the features of the input array.
```
def extract_features(input_arr, model):
    input_shape = (224, 224, 3)
    preprocessed_arr = preprocess_input(input_arr)
    features = model.predict(preprocessed_arr, verbose = 0)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features
```

## Generating Features
This part of the code applies `extract_features` function to each array in the `data_list` generated and stores the features in the `feature_list`.
```
feature_list = []
for i in range(len(data_list)):
    data = data_list[i]
    feature_list.append(extract_features(data, model))
```

## Finding the Nearest Neighbor
I imported `NearestNeighbors` from `sklearn.neighbors` to find the nearest neighbor using cosine similarity and Euclidean distance. 
In this case, both yielded the same result.
```
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(feature_list)
```
or, 
```
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
```

## Finding the Nearest Neighbor of a Certain Interval
The following image is the pattern with a `f_start` of `1530 + 256 * 2.79e-6 * 3015` and an interval of `256 * 2.79e-6`. This pattern would be the 3015th (zero-index) of the `feature_list`. 
```
start = 1530 + 256 * 2.79e-6 * 3015
stop = start + 256 * 2.79e-6
wf.plot_waterfall(f_start=start, f_stop=stop)
```
![Screen Shot 2022-12-04 at 16 53 03](https://user-images.githubusercontent.com/67254464/205526913-48a24cf8-357a-46e0-80f2-f36931abf7c3.png)

```
distances, indices = neighbors.kneighbors([feature_list[3015]])
```
The indices are `[3015, 3014, 5538, 3348, 3981]`. They are ordered in ascending order of their distance from the 3015th pattern. The first one would be the target pattern itself. 

```
# The 1st nearest neighbor except itself
start = 1530 + 256 * 2.79e-6 * 3014
stop = start + 256 * 2.79e-6
wf.plot_waterfall(f_start=start, f_stop=stop)
```
![Screen Shot 2022-12-04 at 17 21 16](https://user-images.githubusercontent.com/67254464/205529101-234a76a4-f2aa-495a-bd47-fa482df31e2a.png)  

```
# The 2nd nearest neighbor except itself
start = 1530 + 256 * 2.79e-6 * 5538
stop = start + 256 * 2.79e-6
wf.plot_waterfall(f_start=start, f_stop=stop)
```
![Screen Shot 2022-12-04 at 17 22 42](https://user-images.githubusercontent.com/67254464/205529219-0a0b0afd-7026-499b-9e73-47c4ad1625c1.png)  

```
# The 3rd nearest neighbor except itself
start = 1530 + 256 * 2.79e-6 * 3348
stop = start + 256 * 2.79e-6
wf.plot_waterfall(f_start=start, f_stop=stop)
```
![Screen Shot 2022-12-04 at 17 23 56](https://user-images.githubusercontent.com/67254464/205529353-add3c1c9-0dfe-4c05-8840-83012b2878ba.png)  

```
# The 4th nearest neighbor except itself
start = 1530 + 256 * 2.79e-6 * 3981
stop = start + 256 * 2.79e-6
wf.plot_waterfall(f_start=start, f_stop=stop)
```
![Screen Shot 2022-12-04 at 17 24 25](https://user-images.githubusercontent.com/67254464/205529401-9800719e-c92c-4e3d-a870-418055803925.png)
