# NeuroLens Streamlit UI prototype

첨부된 NeuroLens 대시보드 시안을 기반으로 만든 프론트엔드 프로토타입입니다.

```powershell
cd E:\해커톤\neurolens_streamlit
python -m pip install -r requirements.txt
streamlit run app.py
```

왼쪽의 **예시 MRI 받기**를 눌러 파일을 내려받은 뒤 **뇌-MRI(T2) 업로드**에 넣으면 전체 결과 화면을 확인할 수 있습니다.

현재는 데모 확률과 판독 문구를 사용합니다. 모델 연동은 `infer()` 함수의 반환값만 교체하면 됩니다.
