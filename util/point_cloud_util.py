import numpy as np
import open3d


def _label_to_colors(labels):
    map_label_to_color = {
        0: [100, 100, 100],  # white
        1: [222,184,135],  # blue
        2: [34,139,34],  # maroon
        3: [255, 0, 255],  # fuchisia
        4: [255,215,0],  # green
        5: [178,34,34],  # red
        6: [0,255,255],  # purple
    }
    return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def _label_to_colors_one_hot(labels):
    map_label_to_color = np.array(
        [
            [100, 100, 100],
            [222,184,135],
            [34,139,34],
            [255, 0, 255],
            [255,215,0],
            [178,34,34],
            [0,255,255],
        ]
    )
    num_labels = len(labels)
    labels_one_hot = np.zeros((num_labels, 7))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return np.dot(labels_one_hot, map_label_to_color).astype(np.int32)


def colorize_point_cloud(point_cloud, labels):
    if len(point_cloud.points) != len(labels):
        raise ValueError("len(point_cloud.points) != len(labels)")
    if len(labels) < 1e6:
        print("_label_to_colors_one_hot used")
        colors = _label_to_colors_one_hot(labels)
    else:
        colors = _label_to_colors(labels)
    # np.testing.assert_equal(colors, colors_v2)
    point_cloud.colors = open3d.Vector3dVector()  # Clear it to save memory
    point_cloud.colors = open3d.Vector3dVector(colors)


def load_labels(label_path):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [int(line) for line in f]
    return np.array(labels, dtype=np.int8)


def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write("%d\n" % label)
