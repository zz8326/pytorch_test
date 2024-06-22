# pip install streamlit-drawable-canvas
# streamlit run app.py
import streamlit as st 
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
import PIL
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize

# 模型載入
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache(allow_output_mutation=True)
def load_model():
    return torch.load('./model.pt', map_location=torch.device('cpu')) #.to(device)

model = load_model()

st.title('手寫阿拉伯數字辨識')

col1, col2 = st.columns(2)
with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=10,
        stroke_color="rgba(0, 0, 0, 1)",
        update_streamlit=True,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas1",
    )

with col2:
    if st.button('辨識'):
        image1 = rgb2gray(rgba2rgb(canvas_result.image_data))
        image_resized = resize(image1, (28, 28), anti_aliasing=True)  
        X1 = image_resized.reshape(1,28,28) 
        # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
        X1 = torch.FloatTensor(1.0-X1) #.to(device)

        # 顯示預測結果
        st.write(f'### 預測結果:{model(X1).argmax(dim=1).item()}')
        st.image(image_resized)

        # PIL transform not work
        # image to a Torch tensor
        # transform = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.Resize([28, 28]),
            # transforms.PILToTensor()
        # ])
        # X1 = transform(PIL.Image.fromarray(canvas_result.image_data[:,:,:3]))
        # X2 = torch.FloatTensor(1.0-X1) #.to(device)

        # st.write(f'### 預測結果:{model(X2).argmax(dim=1).item()}')
        # st.image(X1.numpy().reshape([28, 28]))
