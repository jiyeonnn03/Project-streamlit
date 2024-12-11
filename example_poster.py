import os
import streamlit as st
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

st.title('🎁 제품 홍보 포스터 생성기')
keyword = st.text_input("키워드를 입력하세요.")

if st.button('생성하기🔥'):
    with st.spinner('포스터 생성중'):
        response_text = client.chat.completions.create(
			messages=[
					{
					"role": "system",
					"content": "입력 받은 키워드에 대한 150자 이내의 솔깃한 제품 홍보 문구를 작성해줘.",
				},
				{
					"role": "user",
					"content": keyword,
				}
			],
			model="gpt-4",
		)
        result_text = response_text.choices[0].message.content
        st.success(result_text)
        
        response_image = client.images.generate(
			model="dall-e-3",
			prompt=f'{keyword} 제품을 홍보하기 위한 포스터를 생성하려고 해. 제품 홍보 문구 \'{result_text}\'에 대한 이미지를 생성해줘.',
			size="1024x1024",
			n=1,
			)
        image_url = response_image.data[0].url
        st.image(image_url)