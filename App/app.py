import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import os
import time
class_names=['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheese_plate','cheesecake','chicken_curry','chicken_quesadilla','chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel','filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']

def preprocess(pil_image,scale=False):
    image=np.array(pil_image)
    image=tf.convert_to_tensor(image)
    image = tf.image.resize(image,size=(224,224))
    image=tf.expand_dims(image,axis=0)
    if scale:
        image=image/255.
    return image

@st.cache_resource
def load_model():
    # Get the current directory of the app file
    app_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the model file
    model_path = os.path.join(app_directory, 'efficientnetv2_fine_tuned_79.h5')

    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model



fine_tuned_efficientnetv2=load_model()

def state_initalizer():
    if 'image' not in st.session_state:
        st.session_state['image']='Not Done'
    if 'upload_button' not in st.session_state:
        st.session_state['upload_button']='Not Clicked'
    if 'camera_button' not in st.session_state:
        st.session_state['camera_button']='Not Clicked'

state_initalizer()

st.title("GastronomAI")
st.write("Upload an image of a dish and we will tell you what it is!")

def upload_callback():
    st.session_state.image='Done'

def upload_button_clicked():
    st.session_state.upload_button='Clicked'
    st.session_state.camera_button='Not Clicked'

def camera_callback():
    st.session_state.camera_button='Clicked'
    st.session_state.upload_button='Not Clicked'

columns=st.columns(2)
columns[0].button("Upload an image",on_click=upload_button_clicked)
columns[1].button("Take a photo",on_click=camera_callback)


if st.session_state['camera_button']=='Clicked':
    cam_component=st.empty()
    camera_image=cam_component.camera_input("Take a photo of a dish",on_change=upload_callback)
    if camera_image:
        pil_image = Image.open(camera_image)
        component=st.empty()
        with st.spinner("Classifying..."):
            preproccessed_image=preprocess(pil_image)
            pred_probs=fine_tuned_efficientnetv2.predict(preproccessed_image,verbose=1)
            pred_class=class_names[pred_probs.argmax()]
            if pred_probs.max()>0.5:
                component.write(f"Prediction: {pred_class} with probability {pred_probs.max()*100:.2f}% ")
            else:
                component.write(f"Sorry, we are not sure what this is. Please try another image.")


if st.session_state['upload_button']=='Clicked':
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'],on_change=upload_callback)
    if st.session_state['image']=='Done':
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption='Your Image.', use_column_width=True)
            component=st.empty()
            with st.spinner("Classifying..."):
                preproccessed_image=preprocess(pil_image)
                pred_probs=fine_tuned_efficientnetv2.predict(preproccessed_image,verbose=1)
                pred_class=class_names[pred_probs.argmax()]
                if pred_probs.max()>0.5:
                    component.write(f"Prediction: {pred_class} with probability {pred_probs.max()*100:.2f}% ")
                else:
                    component.write(f"Sorry, we are not sure what this is. Please try another image.")











