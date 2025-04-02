def get_iou(bb1, bb2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0][0], bb2[0][0])
    y_top = max(bb1[0][1], bb2[0][1])
    x_right = min(bb1[1][0], bb2[1][0])
    y_bottom = min(bb1[1][1], bb2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
    bb2_area = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# iou = get_iou([[0, 0], [1000, 1000]], [[10, 10], [1010, 1010]])
# print(iou)

