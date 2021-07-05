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
    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]

    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "*****":
            break
    return decoded_data[:-5]

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

    print('--------------------------------')
    output_image = "encoded_pic1.png"

    decoded_data = decode(output_image)
    print(">> Decoded message before decryption: ", decoded_data)
    print(">> Decoded message after encryption: ", encode_message(decoded_data))

    print('--------------------------------')

    # output_image = "encoded_pic2.png"
    #
    # decoded_data = decode(output_image)
    # print(">> Decoded message before decryption: ", decoded_data)
    # print(">> Decoded message after encryption: ", encode_message(decoded_data))