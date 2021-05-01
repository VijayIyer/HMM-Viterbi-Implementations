#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np
# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image


# main program
#
gt_row = -1
gt_col = -1
if len(sys.argv) == 2:
    input_filename = sys.argv[1]
elif len(sys.argv) == 4:
    (input_filename, gt_row, gt_col) = sys.argv[1:]
else:
    raise Exception("Program requires either 1 or 3 parameters")

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
input_image_hmm = input_image.copy()
input_image_known = input_image.copy()
outputimage_name1 = '_Simple'+input_filename
outputimage_name2 = '_HMM'+input_filename
outputimage_name3 = '_Known'+input_filename
# naive bayes
ridge1 = []
# ridge = [edge_strength.shape[0]/2] * edge_strength.shape[1]
for col in range(edge_strength.shape[1]):
    # normalize_col = sum([i for i in edge_strength[col]])
    # prob_pixel = [0 for _ in edge_strength[col]]
    max_loc = np.argmax([edge_strength[loc][col] for loc in range(edge_strength.shape[0])])
    ridge1.append(max_loc)
# naive bayes
# output answer
imageio.imwrite(outputimage_name1, draw_edge(input_image, ridge1, (255, 0, 0), 5))
input_image.close()
ridge2 = []
# hmm viterbi no co-ord
init_prob = [edge_strength[loc][0]/sum([i for i in edge_strength[1]]) for loc in range(edge_strength.shape[0])]

trans_prob = zeros((edge_strength.shape[0],edge_strength.shape[0]))
delta = [[0 for _ in range(edge_strength.shape[1])] for _ in range(edge_strength.shape[0])]
paths = [[None for _ in range(edge_strength.shape[1])] for _ in range(edge_strength.shape[0])]
for row in range(edge_strength.shape[0]):
    delta[row][0] = init_prob[row]
normalized_row_tran = zeros((edge_strength.shape[0],edge_strength.shape[0]))
for col in range(1, edge_strength.shape[1]):
    transposed_edge = np.transpose(edge_strength)
    for i, pixel1 in enumerate(transposed_edge[col-1]):

        for j, pixel2 in enumerate(transposed_edge[col]):
            if i +1 >= j >= i-1:
                trans_prob[i][j] = 1
            else:
                trans_prob[i][j] = 1/abs(j - i)


        normalized_row_tran = sum([k for k in trans_prob[i]])
        for j, pixel2 in enumerate(transposed_edge[col]):
            trans_prob[i][j] = trans_prob[i][j]/normalized_row_tran

    normalized_col = sum([edge_strength[j][col] for j in range(edge_strength.shape[0])])
    for pixel in range(edge_strength.shape[0]):
        emission_prob = edge_strength[pixel][col]/normalized_col
        delta[pixel][col] = max([delta[j][col-1]*trans_prob[j][pixel] for j in range(edge_strength.shape[0])])*emission_prob
        paths[pixel][col] = np.argmax([delta[j][col-1]*trans_prob[j][pixel] for j in range(edge_strength.shape[0])])
    nomalized_delta = sum([delta[pixel][col] for pixel in range(len(delta))])
    for pixel in range(len(delta)):
        delta[pixel][col] = delta[pixel][col]/ nomalized_delta

final_path = np.argmax([delta[pos][edge_strength.shape[1]-1] for pos in range(edge_strength.shape[0])])
state_path = [final_path]
# backtracking to get the paths
for t in range(edge_strength.shape[1] - 1, 0, -1):
    state_path.append(paths[final_path][t])
    final_path = state_path[-1]
ridge2 = state_path[::-1]

imageio.imwrite(outputimage_name2, draw_edge(input_image_hmm, ridge2, (0, 0, 255), 5))
input_image_hmm.close()


# row co-ord

trans_prob = zeros((edge_strength.shape[0],edge_strength.shape[0]))
delta = [[0 for _ in range(edge_strength.shape[1])] for _ in range(edge_strength.shape[0])]
paths = [[None for _ in range(edge_strength.shape[1])] for _ in range(edge_strength.shape[0])]
for row in range(edge_strength.shape[0]):
    delta[row][0] = init_prob[row]
normalized_row_tran = zeros((edge_strength.shape[0],edge_strength.shape[0]))
for col in range(1, edge_strength.shape[1]):
    transposed_edge = np.transpose(edge_strength)
    for i, pixel1 in enumerate(transposed_edge[col - 1]):

        for j, pixel2 in enumerate(transposed_edge[col]):
            if col == int(gt_col) and j != int(gt_row):
                trans_prob[i][j] = 0
            else:
                if i + 1 >= j >= i - 1:
                    trans_prob[i][j] = 1
                else:
                    trans_prob[i][j] = 1 / abs(j - i)

        normalized_row_tran = sum([k for k in trans_prob[i]])
        for j, pixel2 in enumerate(transposed_edge[col]):
            trans_prob[i][j] = trans_prob[i][j]/normalized_row_tran

    normalized_col = sum([edge_strength[j][col] for j in range(edge_strength.shape[0])])
    for pixel in range(edge_strength.shape[0]):
        emission_prob = edge_strength[pixel][col] / normalized_col
        delta[pixel][col] = max([delta[j][col - 1] * trans_prob[j][pixel] for j in range(edge_strength.shape[0])]) * emission_prob
        paths[pixel][col] = np.argmax([delta[j][col - 1] * trans_prob[j][pixel] for j in range(edge_strength.shape[0])])
    nomalized_delta = sum([delta[pixel][col] for pixel in range(len(delta))])
    for pixel in range(len(delta)):
        delta[pixel][col] = delta[pixel][col] / nomalized_delta
final_path = np.argmax([delta[pos][edge_strength.shape[1]-1] for pos in range(edge_strength.shape[0])])
state_path = [final_path]
#region Backtracking to get back path from stored paths
for t in range(edge_strength.shape[1] - 1, 0, -1):
    state_path.append(paths[final_path][t])
    final_path = state_path[-1]
ridge2 = state_path[::-1]
print(ridge2)
imageio.imwrite(outputimage_name3, draw_edge(input_image_known, ridge2, (0, 255, 0), 5))

#endregion