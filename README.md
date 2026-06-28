# TDA_colab — HAN + GTN + PDGNN 파이프라인 Colab 버전

이종 그래프(ACM)에서 **위상 특징(EPD)** 을 자동으로 뽑아 노드 분류에 쓰는 파이프라인을
Google Colab 에서 끝까지 돌려보는 노트북입니다.

```
GTN(메타패스 자동 발견) → PDGNN(채널별 EPD) → semantic attention fusion → HAN 분류
```

- 코드 본체(패키지·테스트): https://github.com/jjune5/TDA
- 노트북: [`TDA_ACM.ipynb`](TDA_ACM.ipynb)

> **Colab 에서는 SLURM 을 쓰지 않습니다.** 노트북이 Colab GPU 런타임에서 `tda.train.run()`
> 을 직접 호출해 끝까지 실행합니다. (SLURM 배치 스크립트는 클러스터 전용이며 본체 저장소의
> `experiments/` 에만 있습니다 — Colab 에서는 무시하세요.)

## 사용법

1. `TDA_ACM.ipynb` 를 Colab 으로 엽니다
   ([Open in Colab](https://colab.research.google.com/github/jjune5/TDA_colab/blob/main/TDA_ACM.ipynb)).
2. **런타임 → 런타임 유형 변경 → GPU** 로 설정합니다.
3. 셀을 위에서부터 실행합니다. 노트북이 알아서:
   - PyG·gudhi 설치, `jjune5/TDA` 클론·설치
   - ACM(HGB) 자동 다운로드
   - 전체 파이프라인 + baseline(HAN 단독) 실행, test Macro-F1 비교
   - GTN 이 발견한 메타패스 어텐션 / fusion 가중치 출력

## 메모

- 원본 PDGNN 의 `torch_scatter`, persistence image 의 cython(`sg2dgm`) 의존성은 제거돼
  있어(각각 torch 기본 scatter, 순수 numpy 로 대체) Colab 에서 추가 빌드가 필요 없습니다.
- 출력 수치는 단일 시드 실측이며 성능 주장이 아닙니다. 충실도·가정·한계는 본체 저장소의
  `README.md` / `docs/design.ko.md` 를 참고하세요.
