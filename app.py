import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Function to apply the cartoon effect ---
def cartoonify_image(image):
    """
    Applies a cartoon effect to an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The cartoonified image.
    """
    # Convert image to a NumPy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGB to BGR as OpenCV uses BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a median blur to the grayscale image to reduce noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Use adaptive thresholding to detect and create strong edges
    # The output is a binary image (black and white)
    edges = cv2.adaptiveThreshold(
        gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,  # Block size
        2   # C value
    )

    # Apply a bilateral filter to the original color image
    # This filter smooths colors while preserving sharp edges,
    # which is essential for the cartoon look.
    # The result is an image with simplified, flat color regions.
    color_simplified = cv2.bilateralFilter(image, 9, 300, 300)

    # Combine the simplified colors with the black edges
    # We use cv2.bitwise_and to overlay the edges onto the color image
    cartoon = cv2.bitwise_and(color_simplified, color_simplified, mask=edges)

    # Convert the final result back to RGB for Streamlit display
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

    return cartoon

# --- Streamlit Application Layout ---
def main():
    """
    The main function that creates and runs the Streamlit application.
    """
    st.title("Image Cartoonizer")
    st.write("Upload an image to turn it into a cartoon using Python and OpenCV.")
    
    st.markdown("---")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, use_column_width=True)

        # Process the image with the cartoonify function
        st.write("Processing image...")
        cartoon_image = cartoonify_image(image)
        
        with col2:
            st.header("Cartoon Image")
            st.image(cartoon_image, use_column_width=True)

        st.markdown("---")

        # Provide a button to download the processed image
        st.download_button(
            label="Download Cartoon Image",
            data=cv2.imencode('.png', cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="cartoonified_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
