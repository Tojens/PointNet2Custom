import os
import open3d
import numpy as np
import util.provider as provider
from util.point_cloud_util import load_labels

train_file_prefixes = [
    #"7025_66305_no_color","7045_66325_no_color","7005_66575_no_color","7050_66450_no_color","6980_66605_no_color","6840_66560_no_color","6965_66630_no_color","6935_66625_no_color","7025_66310_no_color","7060_66375_no_color","7055_66450_no_color","7030_66515_no_color","6985_66650_no_color","6915_66620_no_color","6950_66645_no_color","6990_66590_no_color","6770_66530_no_color","6985_66595_no_color","7020_66535_no_color","6990_66655_no_color","7025_66535_no_color","6835_66565_no_color","6955_66650_no_color","7055_66415_no_color","7020_66545_no_color","6820_66555_no_color","7055_66350_no_color","6745_66520_no_color","6735_66515_no_color","6990_66585_no_color","7025_66540_no_color","6835_66560_no_color","6815_66550_no_color","7050_66330_no_color","6795_66535_no_color","6915_66615_no_color","6850_66565_no_color","6860_66580_no_color","6995_66575_no_color","7000_66570_no_color","6900_66605_no_color","6880_66585_no_color","6980_66650_no_color","6730_66515_no_color","6890_66595_no_color","6870_66580_no_color","7055_66360_no_color","6905_66610_no_color","7060_66395_no_color","6970_66615_no_color","6995_66590_no_color","6975_66605_no_color","6965_66650_no_color","6780_66530_no_color","6845_66565_no_color","7060_66420_no_color","7025_66300_no_color","6945_66640_no_color","6925_66625_no_color","6975_66650_no_color","6895_66595_no_color","6750_66525_no_color","6820_66550_no_color","6900_66610_no_color","7030_66315_no_color","6790_66540_no_color","6925_66620_no_color","7015_66550_no_color","7025_66525_no_color","6950_66640_no_color","6845_66570_no_color","7060_66385_no_color","6920_66620_no_color","7010_66555_no_color","6755_66520_no_color","6955_66645_no_color","7060_66360_no_color","7055_66445_no_color","6990_66595_no_color","6755_66525_no_color","6980_66610_no_color","7060_66405_no_color","6805_66545_no_color","6760_66530_no_color","6745_66525_no_color","6860_66575_no_color","7050_66460_no_color","6960_66635_no_color","6960_66650_no_color","6930_66630_no_color","6985_66605_no_color","6825_66560_no_color","6810_66550_no_color","7035_66520_no_color","7010_66565_no_color","6855_66570_no_color","6765_66525_no_color","7025_66530_no_color","6880_66595_no_color","7020_66305_no_color","6740_66520_no_color","6885_66595_no_color","7055_66420_no_color","6965_66625_no_color","6935_66635_no_color","7045_66490_no_color","7015_66555_no_color","6725_66515_no_color","6975_66620_no_color","7040_66495_no_color","7050_66485_no_color","6985_66655_no_color","7040_66505_no_color","6980_66600_no_color","6870_66585_no_color","6945_66635_no_color","7055_66460_no_color","7045_66335_no_color","7025_66320_no_color","7055_66430_no_color","7045_66495_no_color","6775_66530_no_color","6760_66525_no_color","7060_66350_no_color","7020_66310_no_color","6885_66590_no_color","6910_66615_no_color","7040_66500_no_color","7000_66575_no_color","7000_66580_no_color","7005_66570_no_color","7040_66330_no_color","7060_66355_no_color","7020_66300_no_color","6995_66655_no_color","6830_66560_no_color","7040_66325_no_color","7045_66480_no_color","6865_66580_no_color","6905_66605_no_color","6960_66640_no_color","7050_66470_no_color","6965_66635_no_color","6975_66610_no_color","7025_66315_no_color","6890_66600_no_color","6850_66570_no_color","7055_66425_no_color","6800_66540_no_color","7055_66455_no_color","7045_66485_no_color","6875_66585_no_color","7050_66455_no_color","7060_66390_no_color","7055_66365_no_color","7060_66415_no_color","6920_66615_no_color","7015_66560_no_color","7050_66475_no_color","7060_66410_no_color","6875_66590_no_color","6970_66620_no_color","6780_66535_no_color","6970_66650_no_color","7055_66335_no_color","6865_66575_no_color","7020_66550_no_color","6770_66525_no_color","6825_66555_no_color","6975_66615_no_color","7030_66525_no_color","7060_66370_no_color","6810_66545_no_color","6795_66540_no_color","6790_66535_no_color","6995_66585_no_color","7050_66335_no_color","7045_66475_no_color","7035_66320_no_color","6800_66545_no_color","6815_66555_no_color","6880_66590_no_color","6785_66535_no_color","6725_66520_no_color","6980_66655_no_color","7035_66515_no_color","7035_66510_no_color","6960_66645_no_color","7045_66330_no_color","7055_66435_no_color","7055_66410_no_color","6970_66630_no_color","7005_66565_no_color","7060_66430_no_color","6775_66535_no_color","7055_66345_no_color","7020_66540_no_color","6995_66580_no_color","6785_66530_no_color","7035_66505_no_color","6895_66605_no_color","7050_66340_no_color","6805_66540_no_color","6940_66630_no_color","7045_66500_no_color","7010_66560_no_color","6730_66520_no_color","6895_66600_no_color","7030_66530_no_color","6850_66575_no_color","7030_66320_no_color","6985_66590_no_color","6955_66640_no_color","6735_66520_no_color","7055_66340_no_color","7050_66465_no_color","6970_66625_no_color","7030_66520_no_color","6915_66610_no_color","6765_66530_no_color","6935_66630_no_color","6910_66610_no_color","7060_66435_no_color","6815_66545_no_color","6965_66640_no_color","7060_66400_no_color","7060_66425_no_color","6875_66580_no_color","7060_66365_no_color","6750_66520_no_color","6840_66565_no_color","7060_66440_no_color","6990_66650_no_color","6830_66555_no_color"

]

validation_file_prefixes = [
    #"7050_66480_no_color","6985_66600_no_color"#,"7055_66440_no_color","6900_66600_no_color","7055_66405_no_color","7015_66545_no_color"
    "6725_66515_no_color",
    "6725_66520_no_color",
    "6730_66515_no_color",
    "6730_66520_no_color",
    "6735_66515_no_color",
    "6735_66520_no_color",
    "6740_66520_no_color",
    "6745_66520_no_color",
    "6745_66525_no_color"
]

test_file_prefixes = [
    #"6940_66635_no_color","6930_66625_no_color","7040_66510_no_color","6855_66575_no_color",
    "6725_66515_no_color",
    "6725_66520_no_color",
    "6730_66515_no_color",
    "6730_66520_no_color",
    "6735_66515_no_color",
    "6735_66520_no_color",
    "6740_66520_no_color",
    "6745_66520_no_color",
    "6745_66525_no_color"

]

all_file_prefixes = train_file_prefixes + validation_file_prefixes + test_file_prefixes

map_name_to_file_prefixes = {
    "train": train_file_prefixes,
    "train_full": train_file_prefixes + validation_file_prefixes,
    "validation": validation_file_prefixes,
    "test": test_file_prefixes,
    "all": all_file_prefixes,
}


class SemanticFileData:
    def __init__(
        self, file_path_without_ext, has_label, use_color, box_size_x, box_size_y
    ):
        """
        Loads file data
        """
        self.file_path_without_ext = file_path_without_ext
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y

        # Load points
        pcd = open3d.read_point_cloud(file_path_without_ext + ".pcd")
        self.points = np.asarray(pcd.points)

        # Load label. In pure test set, fill with zeros.
        if has_label:
            self.labels = load_labels(file_path_without_ext + ".labels")
        else:
            self.labels = np.zeros(len(self.points)).astype(bool)

        # Load colors. If not use_color, fill with zeros.
        if use_color:
            self.colors = np.asarray(pcd.colors)
        else:
            self.colors = np.zeros_like(self.points)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]

    def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
        """
        Get down-sample or up-sample mask to sample points to num_points_per_sample
        """
        # TODO: change this to numpy's build-in functions
        # Shuffling or up-sampling if needed
        if len(points) - num_points_per_sample > 0:
            true_array = np.ones(num_points_per_sample, dtype=bool)
            false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < num_points_per_sample:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[:num_points_per_sample]
        return sample_mask

    def _center_box(self, points):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(points, axis=0)
        shift = np.array(
            [
                box_min[0] + self.box_size_x / 2,
                box_min[1] + self.box_size_y / 2,
                box_min[2],
            ]
        )
        points_centered = points - shift
        return points_centered

    def _extract_z_box(self, center_point):
        """
        Crop along z axis (vertical) from the center_point.

        Args:
            center_point: only x and y coordinates will be used
            points: points (n * 3)
            scene_idx: scene index to get the min and max of the whole scene
        """
        # TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
        box_min = center_point - [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]
        box_max = center_point + [
            self.box_size_x / 2,
            self.box_size_y / 2,
            scene_z_size,
        ]

        i_min = np.searchsorted(self.points[:, 0], box_min[0])
        i_max = np.searchsorted(self.points[:, 0], box_max[0])
        mask = (
            np.sum(
                (self.points[i_min:i_max, :] >= box_min)
                * (self.points[i_min:i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(self.points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask

    def sample(self, num_points_per_sample):
        points = self.points

        # Pick a point, and crop a z-box around
        center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self._extract_z_box(center_point)
        points = points[scene_extract_mask]
        labels = self.labels[scene_extract_mask]
        colors = self.colors[scene_extract_mask]

        sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
        points = points[sample_mask]
        labels = labels[sample_mask]
        colors = colors[sample_mask]

        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self._center_box(points)

        return points_centered, points, labels, colors

    def sample_batch(self, batch_size, num_points_per_sample):
        """
        TODO: change this to stack instead of extend
        """
        batch_points_centered = []
        batch_points_raw = []
        batch_labels = []
        batch_colors = []

        for _ in range(batch_size):
            points_centered, points_raw, gt_labels, colors = self.sample(
                num_points_per_sample
            )
            batch_points_centered.append(points_centered)
            batch_points_raw.append(points_raw)
            batch_labels.append(gt_labels)
            batch_colors.append(colors)

        return (
            np.array(batch_points_centered),
            np.array(batch_points_raw),
            np.array(batch_labels),
            np.array(batch_colors),
        )


class SemanticDataset:
    def __init__(
        self, num_points_per_sample, split, use_color, box_size_x, box_size_y, path
    ):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size_x (int): Defaults to 10. The size of the extracted cube.
        box_size_y (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.num_points_per_sample = num_points_per_sample
        self.split = split
        self.use_color = use_color
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.num_classes = 7
        self.path = path
        self.labels_names = [
            "unlabeled",
            "terrain",
            "vegetation",
            "noise lower",
            "wires",
            "crossbeam",
            "noise upper",
            ]

        # Get file_prefixes
        file_prefixes = map_name_to_file_prefixes[self.split]
        print("Dataset split:", self.split)
        print("Loading file_prefixes:", file_prefixes)

        # Load files
        self.list_file_data = []
        for file_prefix in file_prefixes:
            file_path_without_ext = os.path.join(self.path, file_prefix)
            file_data = SemanticFileData(
                file_path_without_ext=file_path_without_ext,
                has_label=self.split != "test",
                use_color=self.use_color,
                box_size_x=self.box_size_x,
                box_size_y=self.box_size_y,
            )
            self.list_file_data.append(file_data)

        # Pre-compute the probability of picking a scene
        self.num_scenes = len(self.list_file_data)
        self.scene_probas = [
            len(fd.points) / self.get_total_num_points() for fd in self.list_file_data
        ]

        # Pre-compute the points weights if it is a training set
        if self.split == "train" or self.split == "train_full":
            # First, compute the histogram of each labels
            label_weights = np.zeros(7)
            for labels in [fd.labels for fd in self.list_file_data]:
                tmp, _ = np.histogram(labels, range(8))
                label_weights += tmp

            # Then, a heuristic gives the weights
            # 1 / log(1.2 + probability of occurrence)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = 1 / np.log(1.2 + label_weights)
        else:
            self.label_weights = np.zeros(7)

    def sample_batch_in_all_files(self, batch_size, augment=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for _ in range(batch_size):
            points, labels, colors, weights = self.sample_in_all_files(is_training=True)
            if self.use_color:
                batch_data.append(np.hstack((points, colors)))
            else:
                batch_data.append(points)
            batch_label.append(labels)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        if augment:
            if self.use_color:
                batch_data = provider.rotate_feature_point_cloud(batch_data, 3)
            else:
                batch_data = provider.rotate_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def sample_in_all_files(self, is_training):
        """
        Returns points and other info within a z - cropped box.
        """
        # Pick a scene, scenes with more points are more likely to be chosen
        scene_index = np.random.choice(
            np.arange(0, len(self.list_file_data)), p=self.scene_probas
        )

        # Sample from the selected scene
        points_centered, points_raw, labels, colors = self.list_file_data[
            scene_index
        ].sample(num_points_per_sample=self.num_points_per_sample)

        if is_training:
            weights = self.label_weights[labels]
            return points_centered, labels, colors, weights
        else:
            return scene_index, points_centered, points_raw, labels, colors

    def get_total_num_points(self):
        list_num_points = [len(fd.points) for fd in self.list_file_data]
        return np.sum(list_num_points)

    def get_num_batches(self, batch_size):
        return int(
            self.get_total_num_points() / (batch_size * self.num_points_per_sample)
        )

    def get_file_paths_without_ext(self):
        return [file_data.file_path_without_ext for file_data in self.list_file_data]
