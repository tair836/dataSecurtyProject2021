# Tair Shriki 211966379
# Ruth Bracha Cohen 314653320
# Margalit Lionov 316206879
# Ravit Clark 208105270

import cv2
import numpy as np

success = 1

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

def encode_with_canny(input_image, secret_data):

    # read the image and convert it to bits matrix
    image = cv2.imread(input_image)

    # get the image height, width, channel
    height, width, channel = image.shape

    # checking if data can be encoded in image kernel
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print(">> Maximum bytes to encode in your entered image:", n_bytes)

    if len(secret_data) > n_bytes:
        raise ValueError("\n>> Opppps, insufficient bytes, need bigger image to encode or less data\n")

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
    # print(edges)

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
        print("Exit of program...")
        exit()


    # modified cover image using pixel intensity adjustment
    # changing matrix pixels in edges (found by canny algorithm)
    i = 0
    count = 0
    for row in image:
        j = 0
        for pixel in row:
                # checking if this pixel is an edge --> if yes : encrypt data in this pixel
                if edges[i][j] != 0:
                    count += 1
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

    print('>>> Number of pixels found with canny: ', count)

    return image

def encode_without_canny(input_image, secret_data):

    # read the image and convert it to bits matrix
    image = cv2.imread(input_image)

    # get the image height, width, channel
    height, width, channel = image.shape

    # checking if data can be encoded in image kernel
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print(">> Maximum bytes to encode in your entered image:", n_bytes)

    if len(secret_data) > n_bytes:
        raise ValueError("\n>> Opppps, insufficient bytes, need bigger image to encode or less data\n")

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

    # threshold
    thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Threshold binary inverse image in dilate-diff algorithm", thresh)

    # morphology edgeout = dilated_mask - mask
    # morphology dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    cv2.imshow("dilate image", dilate)

    # get absolute difference between dilate and thresh
    diff = cv2.absdiff(dilate, thresh)
    cv2.imshow("diff image", diff)

    # invert
    edges = 255 - diff
    # print(edges)

    # show the image edges on the newly created image window
    cv2.imshow("Image edges detection without canny algorithm", edges)
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
        print("Exit of program...")
        exit()


    # modified cover image using pixel intensity adjustment
    # changing matrix pixels in edges (found by canny algorithm)
    i = 0
    count = 0
    for row in image:
        j = 0
        for pixel in row:
                # checking if this pixel is an edge --> if yes : encrypt data in this pixel
                if edges[i][j] != 0:
                    count += 1

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

    print(">>> Number of pixels found without canny: ", count)

    return image


def encode_message(data):
    # conversion Chart
    conversion_code = {
        # Uppercase Alphabets
        'A': 'Z', 'B': 'Y', 'C': 'X', 'D': 'W', 'E': 'V', 'F': 'U',
        'G': 'T', 'H': 'S', 'I': 'R', 'J': 'Q', 'K': 'P', 'L': 'O',
        'M': 'N', 'N': 'M', 'O': 'L', 'P': 'K', 'Q': 'J', 'R': 'I',
        'S': 'H', 'T': 'G', 'U': 'F', 'V': 'E', 'W': 'D', 'X': 'C',
        'Y': 'B', 'Z': 'A',

        # Lowercase Alphabets
        'a': 'z', 'b': 'y', 'c': 'x', 'd': 'w', 'e': 'v', 'f': 'u',
        'g': 't', 'h': 's', 'i': 'r', 'j': 'q', 'k': 'p', 'l': 'o',
        'm': 'n', 'n': 'm', 'o': 'l', 'p': 'k', 'q': 'j', 'r': 'i',
        's': 'h', 't': 'g', 'u': 'F', 'v': 'e', 'w': 'd', 'x': 'c',
        'y': 'b', 'z': 'a'
    }

    # Creating converted output
    converted_data = ""

    for i in range(0, len(data)):
        if data[i] in conversion_code.keys():
            converted_data += conversion_code[data[i]]
        else:
            converted_data += data[i]

    # Printing converted output
    return converted_data

if __name__ == "__main__":

    print('----------------Encoding message in image using dilate-diff algorithm----------------\n')
    input_image = "12.png"
    output_image = "encoded_pic1.png"
    print("Enter your massage:")
    x = input()
    secret_data = x

    encrypt_secret_data = encode_message(secret_data)

    # encode the data into the image
    encoded_image1 = encode_without_canny(input_image, encrypt_secret_data)

    # save the output image (encoded image)
    cv2.imwrite(output_image, encoded_image1)
    # decode the secret data from the image


    print('----------------Encoding message in image using canny algorithm----------------\n')

    input_image = "12.png"
    output_image = "encoded_pic2.png"
    print("Enter your massage:")
    x = input()
    secret_data = x

    encrypt_secret_data = encode_message(secret_data)

    # encode the data into the image
    encoded_image2 = encode_with_canny(input_image, encrypt_secret_data)

    # save the output image (encoded image)
    cv2.imwrite(output_image, encoded_image2)
    # decode the secret data from the image
