import cv2
import numpy as np



def draw_detections(item, frame):
    for center, scale, handside in zip(item["centers"], item["scales"], item["hands"]):
        x1, y1 = center[0] - scale[0] / 2, center[1] - scale[1] / 2
        x2, y2 = center[0] + scale[0] / 2, center[1] + scale[1] / 2
        color = (0, 255, 0) if handside == 'left' else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def plot_hand(coords_xy, img):
    coords_xy = coords_xy.astype(int)
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])
    colors = np.uint8(colors*255)

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xy[connection[0], :]
        coord2 = coords_xy[connection[1], :]
        cv2.line(img, tuple(coord1), tuple(coord2), color=tuple(color.tolist()), thickness=2)



