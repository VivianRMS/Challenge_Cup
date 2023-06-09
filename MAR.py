from scipy.spatial import distance as dist


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[49], mouth[56])  # 49, 56
    B = dist.euclidean(mouth[51], mouth[54])  # 51, 54

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[58], mouth[62])  # 58, 62

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar

def mouth_upper_ratio(mouth):
    A = dist.euclidean(mouth[50], mouth[60])
    B = dist.euclidean(mouth[60], mouth[64])
    mur = A / B
    return mur

def mouth_lower_ratio(mouth):
    A = dist.euclidean(mouth[55], mouth[64])
    B = dist.euclidean(mouth[60], mouth[64])
    mlr = A / B
    return mlr