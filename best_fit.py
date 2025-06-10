import cv2
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def fit(img, templates, start_percent, stop_percent, threshold):
    img_width, img_height = img.shape[::-1]
    best_location_count = -1
    best_locations = []
    best_scale = 1

    # plt.axis([0, 2, 0, 1])
    # plt.show(block=False)
    #
    # x = []
    # y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 3)]:
        locations = []
        location_count = 0
        for template in templates:
            template = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

            scores = result[result >= threshold].reshape(-1, 1)  # all relevant scores

            if len(scores) < 2:
                continue
            kmeans = KMeans(n_clusters=2, random_state=0).fit(scores)

            # Get cluster with higher mean (strong matches)
            cluster_labels = kmeans.labels_
            means = kmeans.cluster_centers_

            strong_cluster = np.argmax(means)
            adaptive_threshold = np.mean(scores[cluster_labels == strong_cluster])
            result = np.where(result >= adaptive_threshold)
            # import matplotlib.pyplot as plt

            # plt.hist(scores.flatten(), bins=50, color='blue', alpha=0.7)
            # plt.axvline(np.mean(scores[cluster_labels == strong_cluster]), color='red', label='Adaptive Threshold')
            # plt.legend()
            # plt.title("MatchTemplate Score Distribution")
            # plt.xlabel("Score")
            # plt.ylabel("Frequency")
            # plt.show()
            location_count += len(result[0])
            locations += [result]
        # print("scale: {0}, hits: {1}".format(scale, location_count))
        # x.append(location_count)
        # y.append(scale)
        # plt.plot(y, x)
        # plt.pause(0.00001)
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
            # plt.axis([0, 2, 0, best_location_count])
        elif (location_count < best_location_count):
            pass
    # plt.close()
    return best_locations, best_scale