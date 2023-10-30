from helpers import *
from implementations import *
from model import *


def get_label(data_path):
    lbl = []
    with open(data_path, "r") as f:
        lbl = f.readline().rstrip().split(",")

    lbl = lbl[1:]
    l_dict = {}
    for i, l in enumerate(lbl):
        l_dict.update({l: i})
    return lbl, l_dict


def split_data(x, split_y=True):
    indexes1 = np.where(x[:, l_dict["CTELNUM1"]] == -1)
    i1p = np.where(x[:, l_dict["PVTRESD2"]] == -1)
    indexes2 = np.where(x[:, l_dict["CTELENUM"]] == -1)
    i2p = np.where(x[:, l_dict["PVTRESD1"]] == -1)
    result1 = np.setdiff1d(i1p, indexes1)
    result2 = np.setdiff1d(i2p, indexes2)
    print(result1, result2)
    x_splits = [x[indexes1], x[indexes2]]
    if split_y:
        y_splits = [y[indexes1], y[indexes2]]
        return x_splits, y_splits
    return x_splits


def remove_irrelevant_col(x):
    relevant_indexes = []
    for i, col in enumerate(x.T):
        # define the irrelevant columns: 90% of values are 1/-1
        if np.mean(col == 1) >= 0.9 or np.mean(col == -1) >= 0.9:
            continue
        relevant_indexes.append(i)
    return x[:, relevant_indexes], relevant_indexes


def clean_data(x):
    x_clean = x.copy()
    for i, col in enumerate(x_clean.T):
        med = np.median(col[col != -1])
        # col[(col > 10 * mean_value) | (col == -1)] = mean_value
        col[col == -1] = med
    return x_clean


def normalize_features(x):
    """
    Normalize the features using Z-score normalization.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    standardized_x = (x - mean) / std
    return standardized_x


def smote_oversampling(
    x_train, y_train, minority_class, k_neighbors, oversampling_ratio
):
    minority_indices = np.where(y_train == minority_class)[0]

    num_synthetic_samples = int(len(minority_indices) * oversampling_ratio)

    synthetic_samples = []

    for _ in range(num_synthetic_samples):
        random_minority_index = np.random.choice(minority_indices)
        minority_sample = x_train[random_minority_index]

        distances = np.linalg.norm(x_train - minority_sample, axis=1)
        sorted_indices = np.argsort(distances)
        nearest_neighbors_indices = sorted_indices[1 : k_neighbors + 1]

        random_neighbor_index = np.random.choice(nearest_neighbors_indices)
        neighbor_sample = x_train[random_neighbor_index]

        synthetic_sample = minority_sample + np.random.random() * (
            neighbor_sample - minority_sample
        )

        synthetic_samples.append(synthetic_sample)

    x_train_oversampled = np.vstack([x_train, np.array(synthetic_samples)])
    y_train_oversampled = np.hstack(
        [y_train, np.full(num_synthetic_samples, minority_class)]
    )

    return x_train_oversampled, y_train_oversampled


def down_sampling(x_train, y_train, minority_class, downsampling_ratio):
    minority_class_indices = np.where(y_train == minority_class)[0]
    majority_class_indices = np.where(y_train != minority_class)[0]
    num_samples = int(len(minority_class_indices) * downsampling_ratio)
    random_majority_indices = np.random.choice(
        majority_class_indices, num_samples, replace=False
    )
    balanced_indices = np.concatenate([minority_class_indices, random_majority_indices])
    x_train_downsampled = x_train[balanced_indices]
    y_train_downsampled = y_train[balanced_indices]
    return x_train_downsampled, y_train_downsampled


def train(y, x, seed, k, d, lbd):
    k_fold = build_k_fold(y, k, seed)
    loss_tr, loss_te, f1_tr, f1_te, weight = cross_validation(y, x, k_fold, lbd)
    print(
        "extension degree %d, Lambda value %f, train_loss = %f, test_loss = %f, train_f1 = %f, test_f1 = %f"
        % (d, lbd, loss_tr, loss_te, f1_tr, f1_te)
    )
    return weight


def train_rr(y, x, seed, k, d, lbd):
    k_fold = build_k_fold(y, k, seed)
    loss_tr, loss_te, f1_tr, f1_te, weight = cross_validation_rr(y, x, k_fold, lbd)
    print(
        "extension degree %d, Lambda value %f, train_loss = %f, test_loss = %f, train_f1 = %f, test_f1 = %f"
        % (d, lbd, loss_tr, loss_te, f1_tr, f1_te)
    )
    return weight


x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

# train and get the weight
y = y_train
x = x_train[:, :]
x[np.isnan(x)] = -1

lbl, l_dict = get_label("data/dataset/x_train.csv")

x_splits, y_splits = split_data(x)
x_zero, tr_idx0 = remove_irrelevant_col(x_splits[0])
x_one, tr_idx1 = remove_irrelevant_col(x_splits[1])
x_splits = [x_zero, x_one]

x_clean_0 = clean_data(x_splits[0])
x_clean_1 = clean_data(x_splits[1])
x_clean = [x_clean_0, x_clean_1]
assert not np.any(x_clean_1 == -1)

x_0 = normalize_features(x_clean[0])
x_1 = normalize_features(x_clean[1])
x_ext_0 = poly_extension(x_0, 1)
x_ext_1 = poly_extension(x_1, 1)

x_ds_0, y_ds_0 = down_sampling(
    x_ext_0, y_splits[0], minority_class=1, downsampling_ratio=2.0
)
x_ds_1, y_ds_1 = down_sampling(
    x_ext_1, y_splits[1], minority_class=1, downsampling_ratio=3.5
)

weight_0 = train_rr(y_ds_0, x_ds_0, 1000, 5, 1, 1e-8)
weight_1 = train_rr(y_ds_1, x_ds_1, 1000, 5, 1, 1e-5)

# see the f1 score of train set
y_predict_0 = predict_labels(weight_0, x_ext_0)
y_predict_1 = predict_labels(weight_1, x_ext_1)
print(
    "f1_Scores:", f1_score(y_splits[0], y_predict_0), f1_score(y_splits[1], y_predict_1)
)

# get test set
x = x_test[:, :]
x[np.isnan(x)] = -1

# split test set
indexes1 = np.where(x[:, l_dict["CTELNUM1"]] == -1)
indexes2 = np.where(x[:, l_dict["CTELENUM"]] == -1)
assert x.shape[0] == x[indexes1].shape[0] + x[indexes2].shape[0]
x_splits = [x[indexes1], x[indexes2]]

# clean test set
x_zero = x_splits[0][:, tr_idx0]
x_one = x_splits[1][:, tr_idx1]
x_splits = [x_zero, x_one]
x_clean_0 = clean_data(x_splits[0])
x_clean_1 = clean_data(x_splits[1])
x_clean = [x_clean_0, x_clean_1]

# standalization
x_0 = normalize_features(x_clean[0])
x_1 = normalize_features(x_clean[1])
x_ext_0 = poly_extension(x_0, 1)
x_ext_1 = poly_extension(x_1, 1)

# predict
# y_pred = np.zeros((x.shape[0], 1))
y_pred = np.zeros(x.shape[0])
y_pred[indexes1] = predict_labels(weight_0, x_ext_0)
y_pred[indexes2] = predict_labels(weight_1, x_ext_1)

test_ids = np.genfromtxt(
    os.path.join("data/dataset", "x_test.csv"),
    delimiter=",",
    skip_header=1,
    dtype=int,
    usecols=0,
)
create_csv_submission(test_ids, y_pred, "sub.csv")
