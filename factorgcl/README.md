# FactorGCL 구현

논문 [Duan, Wang and Li (2025), *FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual Contrastive Learning for Stock Returns Prediction*, AAAI-25, 173–181](https://doi.org/10.1609/aaai.v39i1.31992) 의 방법론을 논문·부록(supplementary material)에 적힌 그대로 구현한 코드입니다.

주가·거래량 패널 데이터와 종목별 산업(사전 팩터) 정보를 넣으면

1. 부록 "Data Preprocessing" 절차대로 무차원화(dimensionless) 전처리 + 다기간 레이블 생성
2. **캐스케이딩 잔차 하이퍼그래프 구조**(prior beta → hidden beta → individual alpha)로 수익률 예측
3. **시간적 잔차 대조학습**(temporal residual contrastive learning, InfoNCE)으로 은닉 팩터 학습 유도
4. 5년:1년:2년 롤링 재학습 + 논문 지표(IC, ICIR, MSE, F1) 산출
5. TopK 전략 투자 시뮬레이션(AR, IR, RoMaD)

까지 수행합니다.

---

## 1. 논문 ↔ 코드 대응표

| 논문 | 코드 |
|---|---|
| Eq. 1 하이퍼그래프 합성곱 `σ(Dn^-1/2 H W De^-1 Hᵀ Dn^-1/2 e w)` | `model.HyperGCNLayer` |
| 특징 추출기 `φ_feat` (GRU + batch norm, 마지막 시점 은닉상태) | `model.FeatureExtractor` |
| Eq. 5 prior beta 모듈 `φ_prior` | `model.PriorBetaModule` |
| Eq. 6 hidden beta 모듈 `φ_hidden` (은닉 팩터 프로토타입 + soft 하이퍼엣지 `β_h = Sigmoid(e_r·cᵀ)`) | `model.HiddenBetaModule` |
| Eq. 7 individual alpha 모듈 | `model.IndividualAlphaModule` |
| Eq. 8 다기간 예측 `ŷ⁽ˡ⁾ = w_o1 e_p + w_o2 e_h + w_o3 e_α + b_o` | `model.FactorGCL.forward` |
| Eq. 9–10 미래 분기(prior·hidden 파라미터 공유, 과거의 `β`·`β_h` 재사용) | `model.FactorGCL.forward_future` |
| Eq. 11 InfoNCE 대조손실 + 2층 MLP 사영 `p(x)` | `loss.info_nce_loss`, `model.ProjectionHead` |
| Eq. 12–13 목적함수 `L = L_mse + γ·L_CL` | `loss.factorgcl_loss` |
| 부록 전처리(상대가격·상대거래량, 레이블 `(p_{t+Δt+1}−p_{t+1})/p_{t+1}` + 횡단면 표준화) | `data.PanelData`, `data.compute_raw_labels`, `data.standardize_labels` |
| 5년:1년:2년 롤링 학습 | `train.rolling_splits`, `train.run_rolling` |
| IC / ICIR / MSE / F1 | `metrics` |
| TopK 투자 시뮬레이션, AR / IR / RoMaD | `backtest`, `metrics.investment_metrics` |
| Table 2 ablation (`-wo Prior/Hidden/Alpha&CL/CL`) | `ModelConfig(use_prior=…, use_hidden=…, use_alpha=…)`, `TrainConfig(gamma=0)` |
| Table 1 단순 베이스라인(MLP·GRU·TCN·Transformer·ALSTM·HyperGCN) | `baselines` |

논문에 명시된 하이퍼파라미터는 전부 기본값입니다: `H = 32`, `M = 32`, GRU 2층, `T = 60`, `T' = 20`, `Δt = 1, 5, 10, 20`, `τ = 0.1`, `γ = 0.1`, Adam `lr = 1e-3`, 100 epoch, early stopping 20, seed 0, TopK = 30, 보유 10일, 거래비용 0.3%.

## 2. 설치

```bash
pip install torch numpy pandas
```

## 3. 입력 형식

**패널 파일** (csv 또는 parquet): 하루-한 종목이 한 행이며, 부록이 쓰는 6개 필드(`D = 6`)가 필요합니다.

```csv
date,stock,open,high,low,close,vwap,volume
2014-01-02,000001.SZ,10.10,10.35,10.02,10.28,10.21,1523000
2014-01-02,000002.SZ,23.40,23.55,23.10,23.18,23.29,880400
```

**산업 파일** (사전 팩터, 논문에서는 83개 2차 산업): `stock`, `industry` 두 열. 패널에 `industry` 열이 있으면 그것을 써도 됩니다.

```csv
stock,industry
000001.SZ,banks
000002.SZ,real_estate
```

- 종목이 상장 전이거나 거래정지여서 행이 없으면 `NaN`으로 두면 됩니다. 결측이 많은 종목-일자는 해당 날짜 횡단면에서 제외되고(`--max-missing-ratio`), 남은 결측은 0으로 채운 뒤 클리핑됩니다.
- 산업을 모르는 종목은 노출도가 전부 0인 고립 노드가 되며, prior 하이퍼그래프에 기여하지도 영향을 받지도 않습니다.

## 4. 실행

```bash
cd factorgcl

# 설치 확인용: 합성 데이터로 전 과정 실행
python run_factorgcl.py --synthetic --outdir out

# 논문 설정 그대로
python run_factorgcl.py --input panel.csv --industry industry.csv \
    --test-start 2020-01-01 --test-end 2023-06-30 --outdir out

# Table 2 ablation
python run_factorgcl.py --input panel.csv --industry industry.csv --ablation --outdir out

# Table 1 베이스라인
python run_factorgcl.py --input panel.csv --industry industry.csv --model gru --outdir out
```

`out/`에 `predictions.csv`(아웃오브샘플 예측), `metrics.csv`(기간별 IC·ICIR·MSE·F1), `backtest.csv`·`backtest_metrics.csv`(TopK 시뮬레이션), `splits.csv`(롤링 창), `history_window*.csv`(학습 곡선)이 저장됩니다.

파이썬에서 직접 호출할 수도 있습니다.

```python
from factorgcl import (PanelData, PreprocessConfig, ModelConfig, TrainConfig,
                       rolling_splits, run_rolling, evaluate, run_topk_backtest,
                       price_frame_from_panel, backtest_metrics)

panel = PanelData(panel_df, industry, PreprocessConfig())      # T=60, T'=20, Δt=1,5,10,20
splits = rolling_splits(panel.dates, 5, 1, 2, test_start="2020-01-01", test_end="2023-06-30")
predictions, records = run_rolling(panel, splits, ModelConfig(), TrainConfig())

print(evaluate(predictions, panel.config.periods))             # Table 1의 IC/ICIR
frame = run_topk_backtest(predictions, price_frame_from_panel(panel), "pred_10",
                          topk=30, holding=10, cost=0.003)
print(backtest_metrics(frame))                                 # Table 3의 AR/IR/RoMaD
```

주요 옵션:

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--seq-len` / `--future-len` | 과거 `T`, 미래 `T'` 길이 | 60 / 20 |
| `--periods` | 예측 기간 `Δt` (다중 레이블) | 1 5 10 20 |
| `--hidden-size` / `--num-hidden-factors` | `H` / `M` | 32 / 32 |
| `--gamma` / `--temperature` | 대조손실 가중치 `γ`, 온도 `τ` | 0.1 / 0.1 |
| `--no-prior` / `--no-hidden` / `--no-alpha` | ablation 스위치 | 꺼짐 |
| `--train-years` / `--valid-years` / `--test-years` | 롤링 창 길이 | 5 / 1 / 2 |
| `--topk` / `--holding` / `--cost` | TopK 전략 설정 | 30 / 10 / 0.003 |
| `--early-stop-metric` | `ic`(검증 IC 평균) 또는 `loss` | ic |

## 5. 구조 요약

한 번의 최적화 스텝은 **하루치 횡단면 전체**입니다. 하이퍼그래프의 결합행렬은 매 거래일 다시 만들어지고 종목 수 `N`도 날마다 달라지므로, 배치를 날짜 단위로 잡는 것이 논문 구조와 일치합니다.

```
x (N,T,D) ──φ_feat──> e_s ──φ_prior(β)──> e_p          ┐
                       │                                │
                       └─ e_r = e_s − e_p ──φ_hidden──> e_h   ├─ 선형결합 → ŷ (N,L)
                                    │                   │
                                    └─ e_s−e_p−e_h ──> e_α    ┘
                                                        │
x' (N,T',D) ─φ'_feat─> e'_s ─(β, β_h 재사용)─> e'_α ──── InfoNCE (positive: 같은 종목의 과거·미래 α)
```

## 6. 검증

```bash
python test_factorgcl.py      # 또는 pytest test_factorgcl.py
```

40개 테스트가 틀리기 쉬운 지점을 확인합니다: 하이퍼그래프 합성곱이 Eq. 1의 밀집 행렬식과 일치하는지, 고립 노드에서 0으로 나누지 않는지, 잔차 캐스케이드와 Eq. 8의 합 분해가 맞는지, 미래 분기가 prior·hidden 파라미터를 공유하고 과거의 노출도를 재사용하는지, InfoNCE가 정의대로인지, 레이블 공식·횡단면 표준화·무차원화가 부록과 맞는지, 롤링 창이 5:1:2로 놓이고 학습 구간의 레이블이 검증 구간을 넘보지 않는지(embargo), 지표와 TopK 거래비용 회계가 정확한지 등입니다.

합성 데이터(`synthetic.py`)는 산업 팩터 + 은닉 팩터 + 개별 잡음이라는 논문의 구조를 그대로 갖고 있고 팩터 수익률에 자기상관을 주어 실제로 예측 가능한 신호가 존재하도록 만들었습니다. 80종목·700일, 최대 25 epoch 설정에서:

| | IC (Δt=1) | IC (Δt=5) | IC (Δt=10) | MSE (Δt=1) | F1 (Δt=1) |
|---|---|---|---|---|---|
| FactorGCL (γ = 0.1) | 0.0678 | 0.0745 | 0.0857 | 0.9963 | 0.5904 |
| `-wo CL` (γ = 0) | 0.0630 | 0.0678 | 0.0695 | 1.0009 | 0.5320 |

구현이 신호를 실제로 학습하고, 대조학습을 넣었을 때 IC가 올라가는 방향도 논문 Table 2와 같습니다. 합성 데이터에서의 동작 확인일 뿐 논문 수치와의 비교는 아닙니다.

## 7. 논문에 적히지 않아 선택한 부분

논문·부록에 명시되지 않은 지점은 모두 옵션으로 빼 두었고, 기본값은 아래와 같이 골랐습니다.

| 항목 | 논문 서술 | 이 구현의 기본값 |
|---|---|---|
| batch norm 위치 | "GRU with a batch normalization" | 입력 6채널에 적용(`--bn-position input`; `output`/`both`/`none` 선택 가능) |
| 미래 분기의 특징 추출기 | Eq. 10 각주는 `φ'_prior`·`φ'_hidden`만 파라미터 공유라고 명시 | 별도 GRU (`--share-future-encoder`로 공유 가능) |
| 미래 alpha 임베딩 | Eq. 10은 alpha 모듈을 거치지 않은 순수 잔차 | 식 그대로 순수 잔차 (`--future-alpha-module`로 Eq. 7 적용 가능) |
| "clip extreme values" | 임계값 미기재 | 정규화 특징을 ±10으로 클리핑(`--feature-clip`) |
| "drop samples with too many missing values" | 임계값 미기재 | 과거 창 결측 비율 20% 초과 시 제외(`--max-missing-ratio`) |
| 레이블 극단값 처리 | 미기재 | 횡단면 표준화 전 median ± 5·MAD 윈저라이즈(`--label-mad-clip`) |
| 미래 창 무차원화 기준 | "construction ... is similar" | 과거 창과 같은 기준가·평균거래량 사용(`--future-norm anchor`; `window`도 선택 가능) |
| early stopping 기준 | "early stopping set to 20 steps" | 검증 IC 평균(`--early-stop-metric ic`; `loss`도 선택 가능) |
| 거래비용 0.3%의 해석 | 매수/매도 구분 없음 | 매수·매도 각각 0.3%(`--cost-mode per_side`; `round_trip`은 합계 0.3%) |
| 학습/검증 구간 경계 | 미기재 | 학습 구간 끝에서 `max(Δt)+1`일 embargo(레이블이 검증 구간을 넘보지 않도록) |

또한 Table 1의 베이스라인 중 자체 논문이 있는 SFM, GAT, HIST, STHAN-SR, FactorVAE, CI-STHPAN은 재구현하지 않았습니다(`baselines.py`에는 부록이 한 문장으로 설명하는 MLP·GRU·TCN·Transformer·ALSTM·HyperGCN만 포함).

데이터는 논문이 쓴 중국 A주(2014-01-01 ~ 2023-06-30, 5028 종목, 83개 2차 산업)를 그대로 넣어야 논문 수치와 비교할 수 있습니다. 이 저장소에는 해당 데이터가 없으므로 Table 1·3의 수치를 그대로 재현한 것은 아닙니다.
