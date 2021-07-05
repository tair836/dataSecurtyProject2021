# Tair Shriki 211966379
# Ruth Bracha Cohen 314653320
# Margalit Lionov 316206879
# Ravit Clark 208105270

import cv2
import numpy as np


def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")


def encode(input_image, secret_data):

    # read the image and convert it to bits matrix - width of kernel
    image = cv2.imread(input_image)

    # get the image height, width, channel
    height, width, channel = image.shape

    # checking if data can be encoded in image kernel
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print(">> Maximum bytes to encode in your entered image:", n_bytes)

    if len(secret_data) > n_bytes:
        raise ValueError(">> Opppps, insufficient bytes, need bigger image to encode or less data\n")

    #  Mask least 2 bits and find edges
    image = image & 252

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply basic thresholding -- the first parameter is the kernel image
    # we want to threshold, the second value is is our threshold
    # check; if a pixel value is greater than our threshold (in this
    # case, img_height), we set it to be *black, otherwise it is *white*
    (threshold, threshInv) = cv2.threshold(img, height, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("Threshold binary inverse image (in gray) ", threshInv)

    # detection of the edges -> canny algorithm gets the bits img & threshInv value = threshold
    # edges is set of indices of edge pixels
    # detect edges in the image. The parameters control the thresholds
    edges = cv2.Canny(image, 100, 200/int(threshold))
    edges = edges | img


    # show the image edges on the newly created image window
    cv2.imshow("Image edges detection using canny algorithm", edges)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    #################################### encoding #########################################
    #######################################################################################
    print("\n>> Encoding data in image, this process may take a few seconds, please stay with us :)\n ")
    # add stopping criteria
    secret_data += "*****"
    data_index = 0
    # convert data to binary
    binary_secret_data = to_bin(secret_data)

    # size of data to hide
    data_len = len(binary_secret_data)

    # edges check
    if(len(edges)<data_len):
        print("\n>> There are not enough edges for your hidden message\n")


    # modified cover image using pixel intensity adjustment
    # changing matrix pixels in edges (found by canny algorithm)
    i = 0
    for row in image:
        j = 0
        for pixel in row:
                # checking if this pixel is an edge --> if yes : encrypt data in this pixel
                if edges[i][j] != 0:

                    # convert RGB values to binary format
                    r, g, b= to_bin(pixel)

                    # modify the least significant bit (LSB) only if there is still data to store
                    if data_index < data_len :
                        # least significant red pixel bit
                        pixel[0] = int(r[:-1] + binary_secret_data[data_index], 2)
                        data_index += 1
                    if data_index < data_len:
                        # least significant green pixel bit
                        pixel[1] = int(g[:-1] + binary_secret_data[data_index], 2)
                        data_index += 1
                    if data_index < data_len:
                        # least significant blue pixel bit
                        pixel[2] = int(b[:-1] + binary_secret_data[data_index], 2)
                        data_index += 1
                    # if data is encoded, just break out of the loop
                    if data_index >= data_len:
                        break
                    j += 1
        i += 1

    return image

def decode(image_name):
    print(">> Decoding data from image\n")
    # read the image
    image = cv2.imread(image_name)
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    # split by 8-bits
    all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "*****":
            break
    return decoded_data[:-5]

if __name__ == "__main__":
    input_image = "12.png"
    output_image = "encoded_image.png"
    secret_data = "This is a hidden message!!! shhhhhhhhhhhhh!!!"

    # encode the data into the image
    encoded_image = encode(input_image, secret_data)

    # save the output image (encoded image)
    cv2.imwrite(output_image, encoded_image)
    # decode the secret data from the image
    decoded_data = decode(output_image)
    print(">> Decoded message:", decoded_data)
