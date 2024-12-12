import os

import streamlit as st

from openai import OpenAI

api_key = os.environ.get('OPENAI_API_KEY')
print(api_key)
client = OpenAI(api_key = api_key)

st.title('한국어-외국어 번역 프로그램')
language = '영어'

## Checkbox
st.write('번역할 언어를 선택하세요.')
check_chinese = st.checkbox('중국어')
check_japanese = st.checkbox('일본어')
if check_chinese :
    language = '중국어'
elif check_japanese :
    language = '일본어'

messages = [{"role": "system", "content": f"당신은 한글을 외국어로 번역하는 번역가입니다. 입력된 한글 문장을 {language}로 번역한 결과를 제시하세요."}]

input_text = st.text_area('한국어 문장을 입력하세요.')
messages.append({"role": "user", "content": input_text})

if st.button('번역') :
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=messages
    )

    output_text = response.choices[0].message.content

    st.success(f'번역 결과::{output_text}')  # 기본 값으로 출력할 텍스트