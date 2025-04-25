import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import wikipedia
import requests
from io import BytesIO
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .confidence-meter {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img, target_size=(224, 224)):
    # Resize the image
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    return img_array

# Function to get disease information from Wikipedia
def get_disease_info(disease_name):
    try:
        # Clean up disease name for better Wikipedia search
        search_term = disease_name.replace("_", " ")
        
        # Try to get a Wikipedia page
        page = wikipedia.page(f"Plant {search_term}")
        summary = page.summary
        url = page.url
        
        # If we got this far, we found a page
        return {
            "summary": summary,
            "url": url,
            "found": True
        }
    except wikipedia.exceptions.DisambiguationError as e:
        # If we get a disambiguation page, try the first option
        try:
            page = wikipedia.page(e.options[0])
            return {
                "summary": page.summary,
                "url": page.url,
                "found": True
            }
        except:
            return {"found": False}
    except:
        # If no direct plant disease page is found, try a more general search
        try:
            results = wikipedia.search(f"{search_term} plant disease", results=1)
            if results:
                page = wikipedia.page(results[0])
                return {
                    "summary": page.summary,
                    "url": page.url,
                    "found": True
                }
            else:
                return {"found": False}
        except:
            return {"found": False}

# Function to get model input shape
def get_model_input_shape(model):
    # Create a dummy input to determine the expected shape
    try:
        # Get the model's input shape by looking at its inputs
        if hasattr(model, 'inputs') and model.inputs:
            input_tensor = model.inputs[0]
            input_shape = input_tensor.shape.as_list()
            if len(input_shape) >= 3:  # [None, height, width, channels]
                return tuple(input_shape[1:3])  # Return (height, width)
    except:
        pass
    
    # If we can't determine it directly, try to use a dummy input
    try:
        # Start with a standard size and see if it works
        dummy_input = np.zeros((1, 224, 224, 3))
        model.predict(dummy_input, verbose=0)
        return (224, 224)
    except:
        # If 224x224 doesn't work, try other common sizes
        common_sizes = [(128, 128), (160, 160), (176, 176), (192, 192), (224, 224), (256, 256), (299, 299)]
        
        for size in common_sizes:
            try:
                dummy_input = np.zeros((1, size[0], size[1], 3))
                model.predict(dummy_input, verbose=0)
                return size
            except:
                continue
    
    # Default to 224x224 if we couldn't determine
    return (224, 224)

# Function to load class names from file if available
def load_class_names_from_file(file_path="class_names.json"):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.warning(f"Error loading class names: {e}")
        return None

# Main app
def main():
    st.title("🌿 Plant Disease Detection System")
    
    # Sidebar for navigation and options
    with st.sidebar:
        st.header("Settings")
        model_path = "final_plant_disease_model.keras"
        
        # Image size options
        st.subheader("Image Settings")
        image_size_option = st.radio(
            "Input image size", 
            ["Auto-detect from model", "Custom size"]
        )
        
        custom_image_size = (224, 224)  # Default
        if image_size_option == "Custom size":
            col1, col2 = st.columns(2)
            with col1:
                width = st.number_input("Width", min_value=32, max_value=512, value=224, step=32)
            with col2:
                height = st.number_input("Height", min_value=32, max_value=512, value=224, step=32)
            custom_image_size = (width, height)
        
        # Try to load class names from file first
        saved_class_names = load_class_names_from_file()
        
        # Load class labels
        st.subheader("Class Labels")
        
        # Default full list of PlantVillage dataset classes
        default_classes = """Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Blueberry___healthy
Cherry_(including_sour)___Powdery_mildew
Cherry_(including_sour)___healthy
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy
Grape___Black_rot
Grape___Esca_(Black_Measles)
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy
Orange___Haunglongbing_(Citrus_greening)
Peach___Bacterial_spot
Peach___healthy
Pepper,_bell___Bacterial_spot
Pepper,_bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Raspberry___healthy
Soybean___healthy
Squash___Powdery_mildew
Strawberry___Leaf_scorch
Strawberry___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___healthy"""
        
        # Use saved class names if available
        if saved_class_names:
            default_classes = "\n".join(saved_class_names)
            
        labels_input = st.text_area("Class Labels (one per line)", value=default_classes, height=300)
        
        class_labels = [label.strip() for label in labels_input.split('\n') if label.strip()]
        
        # Button to save class names
        if st.button("Save Class Names"):
            try:
                with open("class_names.json", "w") as f:
                    json.dump(class_labels, f)
                st.success("Class names saved successfully!")
            except Exception as e:
                st.error(f"Error saving class names: {e}")
        
        # Display statistics about class labels
        st.info(f"Total classes: {len(class_labels)}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Upload a plant leaf image to detect diseases using a pre-trained deep learning model.")
    
    # Main content area
    col1, col2 = st.columns([1, 1.5])
    
    # Load model first
    # Load model first
    model = None
    if model_path:
        with st.spinner("Loading model..."):
            model = load_keras_model(model_path)
            
            if model is not None:
                st.success("Model loaded successfully!")
                
                # Determine the input size to use.
                if image_size_option == "Auto-detect from model":
                    input_size = get_model_input_shape(model)
                else:
                    input_size = custom_image_size
                
                # Additional model output verification (e.g., checking number of classes)
                output_layer = model.layers[-1]
                num_classes = None
                if hasattr(output_layer, 'get_config') and 'units' in output_layer.get_config():
                    num_classes = output_layer.get_config()['units']
                elif hasattr(model, 'output_shape') and model.output_shape:
                    num_classes = model.output_shape[-1]
                elif hasattr(output_layer, 'weights') and len(output_layer.weights) > 0:
                    num_classes = output_layer.weights[0].shape[-1]
                if num_classes is None:
                    st.warning("Could not determine the number of output classes from the model.")
                    num_classes = len(class_labels)
                elif num_classes != len(class_labels):
                    st.warning(f"⚠️ Model has {num_classes} output classes but you've provided {len(class_labels)} class labels.")
            else:
                st.error("Failed to load model. Please check the file path and format.")
                input_size = (224, 224)  # Default fallback
    else:
        input_size = (224, 224)  # Default fallback
  # Default fallback
    
    with col1:
        st.subheader("Upload Image")
        
        # Option to use sample images or upload
        image_option = st.radio("Choose an option:", ["Upload my own image", "Use a sample image"])
        
        uploaded_file = None
        if image_option == "Upload my own image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
        else:
            # Sample images
            sample_images = {
                "Apple Scab": "https://www.planetnatural.com/wp-content/uploads/2012/12/apple-scab-1.jpg",
                "Corn Common Rust": "https://extension.umn.edu/sites/extension.umn.edu/files/styles/large/public/corn-common-rust.jpg",
                "Healthy Tomato": "https://www.gardeningknowhow.com/wp-content/uploads/2019/09/healthy-tomato-leaves.jpg"
            }
            
            selected_sample = st.selectbox("Select a sample image:", list(sample_images.keys()))
            
            if selected_sample:
                sample_url = sample_images[selected_sample]
                st.image(sample_url, caption=f"Sample: {selected_sample}", width=300)
                
                # Download the sample image
                response = requests.get(sample_url)
                img_bytes = BytesIO(response.content)
                uploaded_file = img_bytes
        
        process_button = st.button("Analyze Image", key="process_button")
    
    # Initialize result containers
    result_container = st.container()
    
    # Process the image when button is clicked
    if process_button and (uploaded_file is not None) and (model is not None):
        with st.spinner("Analyzing the image..."):
            try:
                # Open the image
                if isinstance(uploaded_file, BytesIO):
                    img = image.load_img(uploaded_file)
                else:
                    img = image.load_img(uploaded_file)
                
                # Display the uploaded image
                with col1:
                    st.image(img, caption="Uploaded Image", width=300)
                
                # Preprocess the image
                img_array = preprocess_image(img, target_size=input_size)
                
                # Log shape for debugging
                st.info(f"Processed image shape: {img_array.shape}")
                
                # Make prediction
                with st.spinner("Running prediction..."):
                    prediction = model.predict(img_array)
                
                # Get the predicted class and confidence
                predicted_class_index = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_index]) * 100
                
                # Map index to class label
                if predicted_class_index < len(class_labels):
                    predicted_disease = class_labels[predicted_class_index]
                else:
                    predicted_disease = f"Unknown Class {predicted_class_index}"
                    st.error(f"Model predicted class index {predicted_class_index} but you only have {len(class_labels)} class labels defined. Please update your class labels to match the model's output classes.")
                
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Results card
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"### Detected Disease: {predicted_disease.replace('___', ' - ')}")
                    
                    # Confidence meter
                    st.markdown(f"**Confidence: {confidence:.2f}%**")
                    st.progress(confidence/100)
                    
                    # Top 3 predictions
                    st.markdown("### Top Predictions")
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(prediction[0])[-3:][::-1]
                    
                    # Create bar chart for top predictions
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    top_labels = []
                    for idx in top_indices:
                        if idx < len(class_labels):
                            disease_name = class_labels[idx].replace('___', ' - ')
                            disease_name = disease_name.replace("_", " ")
                            top_labels.append(disease_name)
                        else:
                            top_labels.append(f"Unknown Class {idx}")
                    
                    # Plot horizontal bar chart
                    bars = ax.barh(top_labels, [prediction[0][i] * 100 for i in top_indices], color='green')
                    
                    # Add percentage labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                               ha='left', va='center')
                    
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_title('Top 3 Predictions')
                    
                    st.pyplot(fig)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Get disease information from Wikipedia
                    st.subheader("Disease Information")
                    
                    # Clean disease name for information lookup
                    disease_name = predicted_disease.split('___')[-1]
                    if disease_name.lower() == "healthy":
                        st.success("The plant appears to be healthy! No disease detected.")
                    else:
                        with st.spinner("Fetching disease information..."):
                            disease_info = get_disease_info(disease_name)
                            
                            if disease_info["found"]:
                                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                                st.markdown(f"### About {disease_name.replace('_', ' ')}")
                                st.write(disease_info["summary"])
                                st.markdown(f"[Read more on Wikipedia]({disease_info['url']})")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning(f"Couldn't find specific information about {disease_name.replace('_', ' ')}. Consider searching online for more details.")
                        
                        # Treatment recommendations
                        st.subheader("Management Recommendations")
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        # Generic treatment recommendations based on disease type
                        if "scab" in disease_name.lower():
                            st.markdown("""
                            - **Fungicide Application**: Apply fungicides during the growing season.
                            - **Prune Infected Areas**: Remove and destroy infected leaves and fruits.
                            - **Improve Air Circulation**: Ensure proper spacing between plants.
                            - **Fall Cleanup**: Remove all fallen leaves and debris to reduce overwintering of the fungus.
                            """)
                        elif "rust" in disease_name.lower():
                            st.markdown("""
                            - **Remove Alternate Hosts**: For cedar apple rust, remove cedar trees in the vicinity if possible.
                            - **Fungicide Application**: Apply protective fungicides early in the growing season.
                            - **Plant Resistant Varieties**: Choose rust-resistant cultivars for future plantings.
                            - **Proper Spacing**: Ensure adequate spacing for good air circulation.
                            """)
                        elif "blight" in disease_name.lower():
                            st.markdown("""
                            - **Crop Rotation**: Practice 1-2 year crop rotation with non-host plants.
                            - **Fungicide Treatment**: Apply appropriate fungicides at first sign of disease.
                            - **Resistant Varieties**: Plant disease-resistant varieties when available.
                            - **Remove Debris**: Clear all plant debris after harvest to reduce pathogen survival.
                            """)
                        elif "rot" in disease_name.lower():
                            st.markdown("""
                            - **Sanitation**: Remove and destroy all infected fruits and plant parts.
                            - **Prune Properly**: Maintain open canopy through proper pruning.
                            - **Fungicide Application**: Apply fungicides during fruit development.
                            - **Avoid Injuries**: Handle fruits carefully to prevent wounds that facilitate infection.
                            """)
                        elif "spot" in disease_name.lower():
                            st.markdown("""
                            - **Crop Rotation**: Rotate crops to non-host plants for at least one year.
                            - **Fungicide Application**: Apply protective fungicides at regular intervals.
                            - **Increase Plant Spacing**: Ensure adequate spacing for better air circulation.
                            - **Avoid Overhead Irrigation**: Water at the base of plants to keep foliage dry.
                            """)
                        else:
                            st.markdown("""
                            - **Consult a Professional**: For specific treatment, consult a local agricultural extension office.
                            - **Remove Infected Parts**: Remove and destroy infected plant parts.
                            - **Improve Growing Conditions**: Ensure proper watering, sunlight, and soil conditions.
                            - **Consider Organic Solutions**: Neem oil or copper-based fungicides may help with many plant diseases.
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing the image: {e}")
                st.error("Full error details:")
                st.exception(e)

if __name__ == "__main__":
    main()