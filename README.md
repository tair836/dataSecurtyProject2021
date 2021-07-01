LSBM-Image-Steganography
Simple implementation of Least Significant Bit Matching Image Steganography<br/>

Group members<br/>
Tair Shriki 211966379<br/>
Ruth Bracha Cohen 314653320<br/>
Margalit Lionov 316206879<br/>
Ravit Clark 208105270<br/>


Requirements<br/>
Windows(tested on Windows10x64)<br/>
Python 3<br/>
Numpy<br/>
Cv2<br/>

Code execution<br/>
The following will run through the code execution in steps of inputs to enter and expected output

<br/>Execute python LSBM-algorithm.py in command prompt
1.In main function user is required to enter:<br/>
    input image to "input_image" field<br/>
    output image to "output_image" field<br/>
    secret message to "secret_data" field<br/>
2.The main function will embeed the secret message using LSBM algorithm and will decode the message<br/>
3.During program running "Threshold binary inverse image (in gray)", "Image edges detection using canny algorithm"
will be printed to the screen<br/>
4.At the end of the program image in the entered name of "output_image" will be created automatically in the src path
<br/>
