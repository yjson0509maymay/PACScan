# NeuroLens Streamlit UI prototype

첨부된 NeuroLens 대시보드 시안을 기반으로 만든 프론트엔드 프로토타입입니다.

```powershell
cd E:\해커톤\neurolens_streamlit
python -m pip install -r requirements.txt
streamlit run app.py
```

왼쪽의 **예시 NIfTI 받기**를 눌러 파일을 내려받은 뒤 `.nii` 또는 `.nii.gz` 업로드 영역에 넣고 **분석 시작**을 누르면 됩니다.

한 번의 실행으로 NIfTI 검증, RAS 방향 표준화, 비영 voxel Min-Max 정규화, 56³ 리사이즈가 실제 수행됩니다. 완료 후 `원본 MRI`, `전처리 결과`, `AI 분석`, `XAI 보고서` 뷰를 자유롭게 전환할 수 있습니다.

Streamlit Cloud 버전은 BRAINTENSOR 파이프라인 중 Python만으로 배포 가능한 단계를 연결합니다. FSL BET, ANTs N4 및 MNI 정합은 별도의 연구용 실행환경이 준비되면 연결해야 합니다. AI 분석 수치는 모델 학습 완료 전까지 데모입니다.

현재는 데모 확률과 판독 문구를 사용합니다. 모델 연동은 `infer()` 함수의 반환값만 교체하면 됩니다.
