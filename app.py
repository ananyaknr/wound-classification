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

dscrp_map1 = {"abrasion wound": "Rough surfaces, fingernail scratches, bite marks",
            "bruises wound": "Hand(slap mark/grab mark), belts, bats, other blunt objects",
            "burn wound": "Heat source (thermal, chemical, electrical contact)",
            "cut wound": "Sharp object (knife, scalpel, glass)",
            "laceration wound": "Object with a rough edge (pipe wrench, bricks)",
            "Stab_wound": "Sharp and pointed object (knife, screwdriver, ice pick)",
            }

dscrp_map2 = {"abrasion wound": "Scraped away superficial layers of the epidermis, usually shallow, may bleed slightly.",
            "bruises wound": "Damaged blood vessels below skin causing discoloration (red, blue, purple), swelling, pain, no broken skin.",
            "burn wound": "Damaged or destroyed skin layers, varies in depth based on how deep the skin damage is.",
            "cut wound": "Clean, straight-edged wound, minimal no tissue bridging.",
            "laceration wound": "Tear produced by blunt force trauma with abraded or crushed skin edges, and incomplete separation of stronger tissue elements. Blood vessels and nerves can be exposed.",
            "Stab_wound": "Deep, usually deeper than their cutaneous length, narrow puncture wound, possible internal bleeding depending on depth and location.",
            }

st.title('Wound Classification')
st.write("This app uses a machine learning model to help classify wound images.")

learn_inf = load_learner(' wound-classification\wound classifier.pkl')

def load_image(image_file):
    img = Image.open(image_file)
    return img

# sample image selection
sample_folder = 'sample_images'
image_fnames = [f for f in os.listdir(sample_folder) if f.endswith((".jpg", ".jpeg", ".png",".webp"))]
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
    pred_prob = float(max(prob))*100
    display_pred_class = display_name_map.get(pred_class, pred_class)    
    st.write("Prediction: " + "{:.2f}".format(pred_prob) +"% "+display_pred_class)

    class_weapon = dscrp_map1.get(pred_class, pred_class)    
    st.write("Possible weapons:", class_weapon)

    class_char = dscrp_map2.get(pred_class, pred_class)    
    st.write("Characteristics:", class_char)

    st.write("Note: Characteristics can vary depending on the severity of the wound and the specific weapon used.")
else:
    st.write("No image chosen. Please select or upload a wound image for classification.")
