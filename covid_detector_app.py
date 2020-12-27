import streamlit as st
import altair as alt
import numpy as np
import cv2
import pandas as pd
import os, urllib
import requests
import zipfile
import tensorflow as tf
from urllib.parse import urlencode
from keras.models import model_from_json
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from matplotlib import cm


# External files to download.
EXTERNAL_DEPENDENCIES = {
    "model_weights": {
        "url": "https://yadi.sk/d/4NhxEEiK1tw4YA",
        "cfg": "https://raw.githubusercontent.com/alinazemlev/MADE-ML/main/model_dense.json",
        "output": "weights.hdf5",
        "output_cfg": "cfg.json",
    },
    "images": {
        "zip": "https://yadi.sk/d/WVks76Ew_-NB1Q",
        "save": "images.zip",
        "csv": "https://raw.githubusercontent.com/alinazemlev/MADE-ML/main/images_update.csv.gz"
    },
    "instructions": {
        "readme": "https://raw.githubusercontent.com/alinazemlev/MADE-ML/main/instructions.md"
    }
}

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

HEADER = "X-ray image"
HEADER_MASK = "Grad CAM image"


@st.cache(show_spinner=False)
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


@st.cache(show_spinner=False)
def loss(output):
    return (output[0][1])


@st.cache(show_spinner=False)
def create_heatmap(model, image):
    gradcam = Gradcam(model,
                      model_modifier=model_modifier,
                      clone=False)
    cam = gradcam(loss,
                  image,
                  penultimate_layer=-1,  # model.layers number
                  )
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)

    return heatmap


@st.cache(show_spinner=False)
def create_model(cfg, output_cfg, weights):
    if os.path.exists(output_cfg):
        json_file = open(output_cfg, 'r')
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        json_file.close()
    else:
        loaded_model_json = get_file_content_as_string(cfg)
        json_file = open(output_cfg, 'w')
        json_file.write(loaded_model_json)
        json_file.close()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    return model


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    response = urllib.request.urlopen(path)
    return response.read().decode("utf-8")


@st.cache(show_spinner=False)
def download_file(path):
    final_url = base_url + urlencode(dict(public_key=path))
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)
    return download_response


@st.cache(show_spinner=False)
def preprocess_images(path_images, name_save_images):
    if os.path.exists(name_save_images.split(".")[0]):
        return
    downloads = download_file(path_images)
    with open(name_save_images, "wb") as fout:
        fout.write(downloads.content)

    with zipfile.ZipFile(name_save_images, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames_for_altair(metadata, column):
    df = metadata["label"].apply(lambda x: 2 if x == "Covid" else 1)
    df = pd.DataFrame(df.values, columns=[column], index=metadata.index)
    return df.reset_index()

def frame_selector_ui(metadata):
    st.sidebar.markdown("# Frame")

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(metadata) - 1, 0)
    name, label = metadata.loc[selected_frame_index]
    selected_frame_df = get_selected_frames_for_altair(metadata, "Covid/Non Covid")

    chart = alt.Chart(selected_frame_df).mark_area().encode(
         alt.X("index:Q"),
         alt.Y("Covid/Non Covid:Q")).properties(width=270, height=200)

    selected_frame_df = pd.DataFrame({"index": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x="index")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    return selected_frame_index, name, label


def draw_image(image, header, description):
    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image, use_column_width=True)

def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    return confidence_threshold


def run_classification(model, path_images, name_save_images, path_metadata):
    @st.cache(show_spinner=False)
    def load_metadata(url):
        return pd.read_csv(url, index_col=0)

    @st.cache(show_spinner=False)
    def load_image(path):
        img = cv2.imread(path)
        img_copy = img / 255.0
        img_copy = np.reshape(img_copy, (1, 224, 224, 3))
        return img, img_copy

    preprocess_images(path_images, name_save_images)
    metadata = load_metadata(path_metadata)
    index, name, label = frame_selector_ui(metadata)
    image_to_show, image = load_image(name_save_images.split(".")[0] + "/" + name)
    confidence_threshold = object_detector_ui()

    pred = model.predict(image)
    if pred[0, 1] > confidence_threshold:
        draw_image(image_to_show, HEADER,
                   "**Human-annotated data** (Frame `%i`) (label `%s`) " % (index, label))

        pred_label = "Covid"
        heatmap = create_heatmap(model, image)
        fin = cv2.addWeighted(heatmap[0], 0.7, image_to_show, 0.3, 0)
        draw_image(fin, HEADER_MASK,
                   "**Covid-detector Model** (label `%s`) (confidence `%3.3f`)" % (pred_label, pred[0, 1]))
    else:
        pred_label = "Non-Covid"
        draw_image(image_to_show, HEADER,
                   "**Human-annotated data** (Frame `%i`) (label `%s`)"% (index, label))
        draw_image(image_to_show, HEADER,
                   "**Covid-detector Model** (label `%s`) (confidence `%3.3f`)" % (pred_label, pred[0, 1]))



@st.cache(show_spinner=False)
@st.cache(suppress_st_warning=True)
def run_classify_single_image(image, model):
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    @st.cache(show_spinner=False)
    def reshape_image(img):
        img_copy = img / 255.0
        img_copy = np.reshape(img_copy, (1, 224, 224, 3))
        return img_copy

    image = reshape_image(opencv_image)
    pred = model.predict(image)
    if pred[0, 1] > 0.5:
        pred_label = "Covid"
        heatmap = create_heatmap(model, image)
        fin = cv2.addWeighted(heatmap[0], 0.5, opencv_image, 0.5, 0)
        draw_image(fin, HEADER_MASK,
                   "**Covid-detector Model** (label `%s`) (confidence `%3.3f`)" % (pred_label, pred[0, 1]))
    else:
        pred_label = "Non-Covid"
        draw_image(opencv_image, HEADER,
                   "**Covid-detector Model** (label `%s`) (confidence `%3.3f`)" % (pred_label, pred[0, 1]))


# This file downloader demonstrates Streamlit animation.
def show_file(file_path, name, name_to_save):
    # These are handles to two visual elements to animate.
    if os.path.exists(name_to_save):
        return
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % name)
        progress_bar = st.progress(0)
        r = download_file(file_path)
        with open(name_to_save, "wb") as output_file:
            size = int(r.headers['Content-Length'].strip())
            bytes = 0
            for buf in r.iter_content(1024):
                if buf:
                    output_file.write(buf)
                    bytes += len(buf)
                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f KB)" %
                                            (name, bytes / 1024, size / 1024))
                    progress_bar.progress(min(bytes / size, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    readme_text = st.markdown(get_file_content_as_string(EXTERNAL_DEPENDENCIES["instructions"]["readme"]))
    show_file(EXTERNAL_DEPENDENCIES["model_weights"]["url"], "weights",
              EXTERNAL_DEPENDENCIES["model_weights"]["output"])
    # create model
    model = create_model(EXTERNAL_DEPENDENCIES["model_weights"]["cfg"],
                         EXTERNAL_DEPENDENCIES["model_weights"]["output_cfg"],
                         EXTERNAL_DEPENDENCIES["model_weights"]["output"])

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run image classification",
                                     "Run loading and classifying image"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run image classification or Run loading and classifying image".')
    elif app_mode == "Run image classification":
        readme_text.empty()
        run_classification(model, EXTERNAL_DEPENDENCIES["images"]["zip"],
                           EXTERNAL_DEPENDENCIES["images"]["save"],
                           EXTERNAL_DEPENDENCIES["images"]["csv"])
    elif app_mode == "Run loading and classifying image":
        readme_text.empty()
        uploaded_file = st.file_uploader("Choose a image file", type="jpg")
        if uploaded_file is not None:
            run_classify_single_image(uploaded_file, model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
