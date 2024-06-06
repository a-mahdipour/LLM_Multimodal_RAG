'''
L3: Large Multimodal Models (LMMs)
Note (Kernel Starting): This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.
'''

# !pip install google-generativeai

# Note: don't forget to set up your GOOGLE_API_KEY to use the Gemini Vision model in the env file.
%env GOOGLE_API_KEY=****

import warnings
warnings.filterwarnings('ignore')

## Load environment variables and API keys
import os
from dotenv import load_dotenv, find_dotenv
​
_ = load_dotenv(find_dotenv()) # read local .env file
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')



# Set the genai library
import google.generativeai as genai
from google.api_core.client_options import ClientOptions
​
genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest",
        client_options=ClientOptions(
            api_endpoint=os.getenv("GOOGLE_API_BASE"),
        ),
)


## Note: learn more about GOOGLE_API_KEY to run it locally.

import textwrap
import PIL.Image
from IPython.display import Markdown, Image
​
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
​
## Function to call LMM (Large Multimodal Model).
def call_LMM(image_path: str, prompt: str) -> str:
    # Load the image
    img = PIL.Image.open(image_path)
​
    # Call generative model
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
​
    return to_markdown(response.text)  


## Analyze images with an LMM
# Pass in an image and see if the LMM can answer questions about it
Image(url= "SP-500-Index-Historical-Chart.jpg")
# Use the LMM function
call_LMM("SP-500-Index-Historical-Chart.jpg", 
    "Explain what you see in this image.")
Analyze a harder image
Try something harder: Here's a figure we explained previously!
Image(url= "clip.png")
call_LMM("clip.png", 
    "Explain what this figure is and where is this used.")

'''
Decode the hidden message
Access Utils File and Helper Functions: To access the files for this notebook, 1) click on the "File" option on 
the top menu of the notebook and then 2) click on "Open". For more help, please see the "Appendix - Tips and Help" Lesson.
'''

Image(url= "blankimage3.png")
# Ask to find the hidden message
call_LMM("blankimage3.png", 
    "Read what you see on this image.")


# How the model sees the picture!
# You have to be careful! The model does not "see" in the same way that we see!

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
​
image = imageio.imread("blankimage3.png")
​
# Convert the image to a NumPy array
image_array = np.array(image)
​
plt.imshow(np.where(image_array[:,:,0]>120, 0,1), cmap='gray');


# Create a hidden text in an image
def create_image_with_text(text, font_size=20, font_family='sans-serif', text_color='#73D955', background_color='#7ED957'):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(background_color)
    ax.text(0.5, 0.5, text, fontsize=font_size, ha='center', va='center', color=text_color, fontfamily=font_family)
    ax.axis('off')
    plt.tight_layout()
    return fig
# Modify the text here to create a new hidden message image!
fig = create_image_with_text("Hello, world!") 
​
# Plot the image with the hidden message
plt.show()
fig.savefig("extra_output_image.png")
# Call the LMM function with the image just generated
call_LMM("extra_output_image.png", 
    "Read what you see on this image.")


# It worked!, now plot the image decoding the message.
image = imageio.imread("extra_output_image.png")
​
# Convert the image to a NumPy array
image_array = np.array(image)
​
plt.imshow(np.where(image_array[:,:,0]>120, 0,1), cmap='gray');
