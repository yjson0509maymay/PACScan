# NeuroLens Streamlit UI prototype

첨부된 NeuroLens 대시보드 시안을 기반으로 만든 프론트엔드 프로토타입입니다.

```powershell
cd E:\해커톤\neurolens_streamlit
python -m pip install -r requirements.txt
streamlit run app.py
```

왼쪽의 **예시 DICOM 폴더 받기**를 눌러 ZIP을 내려받아 압축을 푼 뒤, `환자 T2 MRI DICOM 폴더 선택`에서 압축을 푼 폴더를 선택하고 **분석 시작**을 누르면 됩니다.

폴더 안의 DICOM을 `SeriesInstanceUID`로 분류하고 `SeriesDescription`/`ProtocolName`에서 T2 시리즈를 자동 선택합니다. 한 번의 실행으로 DICOM→3D NIfTI 변환, NIfTI 검증, RAS 방향 표준화, 비영 voxel Min-Max 정규화, 56³ 리사이즈가 실제 수행됩니다. 완료 후 `원본 MRI`, `전처리 결과`, `AI 분석`, `XAI 보고서` 뷰를 자유롭게 전환할 수 있습니다.

Streamlit Cloud 버전은 BRAINTENSOR 파이프라인 중 Python만으로 배포 가능한 단계를 사용합니다. 로컬 버전은 `E:\해커톤\BRAINTENSOR`의 `preparing_ref21order_v1.py`를 직접 불러와 FSL BET → ANTsPy/PD25 정합 → Min-Max 정규화 → 56³ 리사이즈를 실행합니다. N4 bias correction은 사용하지 않습니다. AI 분석 수치는 모델 학습 완료 전까지 데모입니다.

현재는 데모 확률과 판독 문구를 사용합니다. 모델 연동은 `infer()` 함수의 반환값만 교체하면 됩니다.

## 로컬 실제 전처리

두 저장소를 같은 상위 폴더에 나란히 둡니다. PACScan은 실행 시 형제 폴더의
`BRAINTENSOR/01_Preprocessing/스크립트/preparing_ref21order_v1.py`를 자동 탐색합니다.

```text
E:\해커톤\
├─ PACScan\
└─ BRAINTENSOR\
```

새 PC에서는 PACScan의 설치 스크립트로 두 저장소와 Python 의존성을 준비할 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -WorkspaceRoot E:\해커톤
```

```powershell
python -m pip install -r requirements-local.txt
streamlit run app.py
```

기본 환경 경로는 다음 환경변수로 변경할 수 있습니다.

- `PACSCAN_BRAINTENSOR_SCRIPT`: BRAINTENSOR v1 스크립트 경로
- `PACSCAN_PD25_ATLAS`: PD25 아틀라스 NIfTI 경로
- `PACSCAN_WSL_DISTRO`: FSL이 설치된 WSL 배포판(기본 `Ubuntu-24.04`)

실행 결과와 단계별 시간은 `results/runs/<run_id>/`에 자동 저장됩니다. 의료영상과
환자별 실행 기록은 `.gitignore`로 GitHub 업로드에서 제외됩니다.
