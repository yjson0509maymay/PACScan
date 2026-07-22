# ref21order_v1 전처리 결과 검증 기록

## 검증 목적

기존 전처리 산출물이 PACScan 시연에서 단계별 실제 결과로 사용 가능한지 확인한다.

## 확인 일자

2026-07-22

## 원본 위치

로컬 보관 경로: `E:\전처리_ref21order_v1\전처리_ref21order`

> 이 경로와 의료영상 파일은 로컬 전용이며 GitHub에 업로드하지 않는다.

## 실행 매니페스트

| 항목 | 값 |
|---|---|
| 실행 스크립트 | `preparing_ref21order.py` |
| timestamp | `20260719_130725` |
| atlas | `PD25-T1MPRAGE-template-1mm.nii.gz` |
| normalization | `minmax` |
| N4 bias correction | `false` |
| target shape | `56×56×56` |
| BET fraction | `0.5` |
| 전체 입력 | 304 |
| success | 292 |
| skipped | 11 |
| failed | 1 |
| elapsed | 7,074.5초 |

## 파일 시스템 실측

| 산출물 | 확인 개수 |
|---|---:|
| 최종 `.nii.gz` | 303 |
| `_work` 환자 폴더 | 303 |
| `*_01_bet.nii.gz` | 303 |
| `*_02_reg.nii.gz` | 302 |
| `*_03_norm.nii.gz` | 302 |

## 수량 해석

매니페스트의 `success 292 + skipped 11 = 303`은 최종 파일 303개와 일치한다. 전체 입력 304개 중 1개가 실패한 기록과도 일치한다.

정합 및 정규화 중간 파일은 각각 302개로 최종 파일보다 1개 적다. 다음 가능성을 확인해야 한다.

- 이전 실행에서 이미 생성된 최종 파일이 있어 중간 단계를 건너뜀
- 중간 파일이 수동으로 정리됨
- 별도 파라미터 조정 또는 재처리 경로 사용

## 시연 대상 선정 조건

시연 환자는 다음 파일이 모두 존재해야 한다.

```text
원본 NIfTI
<sample>_01_bet.nii.gz
<sample>_02_reg.nii.gz
<sample>_03_norm.nii.gz
최종 <sample>.nii.gz
```

또한 다음을 확인한다.

- shape 및 affine을 nibabel로 정상 로드할 수 있음
- NaN/Inf가 없음
- BET 결과에서 과도한 뇌 조직 손실이 없음
- 정합 결과가 atlas 공간에서 심하게 벗어나지 않음
- 정규화 결과가 빈 영상이 아님
- 개인정보가 파일명 또는 헤더에 포함되지 않음

## 시연 화면 매핑

| 화면 | 실제 산출물 |
|---|---|
| 원본 MRI | 원본 NIfTI |
| BET 결과 | `_01_bet.nii.gz` |
| 정합 결과 | `_02_reg.nii.gz` |
| 정규화 결과 | `_03_norm.nii.gz` |
| 최종 전처리 | 최종 56³ `.nii.gz` |
| AI 분석 | 모델 완료 전 데모 |
| XAI 보고서 | 모델 완료 전 데모 |

## 현재 결론

`ref21order_v1` 결과는 PACScan 시연의 실제 전처리 비교 자료로 사용할 수 있다. N4는 의도적으로 사용하지 않으며 화면과 문서에서 N4 완료 상태를 표시하지 않는다.

## 다음 검증

1. 중간 파일이 모두 존재하는 대표 환자 선정
2. 원본 파일 위치 매칭
3. 단계별 중앙 슬라이스 PNG 생성
4. 단계별 intensity 통계 및 shape 기록
5. 시연 자산으로 복사하기 전 DICOM/NIfTI 비식별화 확인

