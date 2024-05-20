import streamlit as st
import pathlib
from fastbook import *
from PIL import Image

# temp = pathlib.PosixPath   
# pathlib.PosixPath = pathlib.WindowsPath

display_name_map = {"abrasion wound": "Abrasion",
                    "bruises wound": "Bruise/Contusion",
                    "burn wound": "Burn wound",
                    "cut wound": "Cut/Incised wound",
                    "laceration wound": "Laceration",
                    "Stab_wound": "Stab wound",
                    }

dscrp_map = {"abrasion wound": "(Description for abrasion class)",
            "bruises wound": "(Description for bruises class)",
            "burn wound": "(Description for burn class)",
            "cut wound": "(Description for cut class)",
            "laceration wound": "(Description for laceration class)",
            "Stab_wound": "(Description for stab class)",
            }

st.title('Wound Classification')
st.write("This app uses a machine learning model to help classify wound images.")

learn_inf = load_learner('upsampled-ENS-model.pkl')

def load_image(image_file):
    img = Image.open(image_file)
    return img

# sample image selection
sample_folder = 'sample_images'
image_fnames = [f for f in os.listdir(sample_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
selected_file = st.selectbox("Select a sample image (optional)", [""] + image_fnames)

# upload your own image 
uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])

# display image 
if uploaded_file is not None:
    image_file = uploaded_file
elif selected_file:
    selected_image_path = os.path.join(sample_folder, selected_file)
    image_file = selected_image_path
else:
    image_file = None 

# prediction
if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption="Chosen Image")
    pred_class, class_num, prob = learn_inf.predict(image)

    display_pred_class = display_name_map.get(pred_class, pred_class)    
    st.write("Prediction:",  display_pred_class)
    # st.write(str(float(max(prob))))

    class_description = dscrp_map.get(pred_class, pred_class)    
    st.write("Description:", class_description)
else:
    st.write("No image chosen. Please select or upload a wound image for classification.")
