import streamlit as st
import table_assembly
import pickle
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

st.title('Конвертация таблицы с фотографии в csv-файл')

img = st.file_uploader(
    'Загрузите сюда свою фотографию',
    type = ['png', 'jpg', 'psd', 'bmp']
    )

if img:
    detection_model = pickle.load(open('./detection_model.pkl', 'rb'))
    recognition_model = pickle.load(open('./recognition_model.pkl', 'rb'))
    model = ocr_predictor(
    det_arch=detection_model,
    reco_arch=recognition_model
        )
    st.header('Ваше изображение')
    st.image(img)
    file = img.read()
    image_result = open('./img.jpg', 'wb')
    image_result.write(file)
    image_result.close()

    image = DocumentFile.from_images('./img.jpg')
    result = model(image)
    jsn = result.pages[0].export()
    lst = []
    for b in jsn['blocks']:
        for l in b['lines']:
            for w in l['words']:
                lst.append(w['value'])

    df = table_assembly.assembly(lst)
    st.header('Распознанная информация')
    df

    st.download_button(
        label="Скачать csv-таблицу",
        data=df.to_csv(encoding='cp1251'),
        file_name='table.csv',
        mime='csv',
        )
