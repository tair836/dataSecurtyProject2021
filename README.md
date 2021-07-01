LSBM-Image-Steganography
Simple implementation of Least Significant Bit Matching Image Steganography<br/>

Group members<br/>
Tair Shriki 211966379
Ruth Bracha Cohen 314653320
Margalit Lionov 316206879
Ravit Clark 208105270


Requirements
Windows(tested on Windows10x64)
Python 3
Numpy
Cv2

Code execution
The following will run through the code execution in steps of inputs to enter and expected output

Execute python LSBM-algorithm.py in command prompt
1.In main function user is required to enter:
    input image to "input_image" field
    output image to "output_image" field
    secret message to "secret_data" field
2.The main function will embeed the secret message using LSBM algorithm and will decode the message
3.During program running "Threshold binary inverse image (in gray)", "Image edges detection using canny algorithm"
will be printed to the screen
4.At the end of the program image in the entered name of "output_image" will be created automatically in the src path
