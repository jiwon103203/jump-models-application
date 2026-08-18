# 롤링 재추정 파이프라인 (Rolling Refit Pipeline)

csv/엑셀 파일 하나(날짜·종가·무위험금리)를 넣으면 논문 [Shu, Yu and Mulvey (2024), *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach*](https://doi.org/10.1057/s41260-024-00376-x)의 절차대로

1. 초과수익률 계산 →
2. 지수가중이동(EWM) downside deviation·Sortino ratio 피처 생성 (+ 원하면 **사용자 커스텀 변수** 추가) →
3. **6개월마다(1월·7월 첫 영업일) 3000거래일 학습창으로 점프 모델 재추정** + 재추정 사이 구간은 온라인 추론 →
4. 0/1 전략 백테스트(거래비용·거래지연 반영) + **거래 지연 로버스트니스 표(논문 Table 5)** →
5. 선택적으로 **HMM 벤치마크(논문 §3.3)** 와 성과 비교

까지 수행하고 결과 csv와 그림을 저장합니다.

---

## 1. 입력 파일 형식

세 개의 열이 필요하며, 열 이름은 자동 인식됩니다(대소문자·공백·단위 표기 무시).

| 역할 | 인식되는 열 이름 예시 |
|---|---|
| 날짜 | `날짜`, `일자`, `기준일자`, `date` |
| 종가 | `종가`, `수정종가`, `종가 (원)`, `close`, `price` |
| 무위험금리 | `무위험금리`, `무위험금리(%)`, `무위험수익률`, `rf`, `risk free rate` |

```csv
날짜,종가,무위험금리
1990-01-02,359.69,7.83
1990-01-03,358.76,7.89
```

- 자동 인식이 실패하면 `--date-col/--close-col/--rf-col`로 직접 지정하면 됩니다.
- csv 인코딩은 `utf-8` / `cp949` / `euc-kr`을 자동으로 시도합니다. 엑셀(`.xlsx`, `.xls`)은 `--sheet`로 시트를 고를 수 있습니다.
- 천 단위 콤마(`1,234.5`), 퍼센트 기호(`4.25%`)가 섞여 있어도 숫자로 변환합니다.
- **무위험금리는 기본적으로 연율 %(예: `4.25` = 연 4.25%)로 해석**하고 `연율/100/252`로 일간화합니다. 다른 형식이면 `--rf-unit annual_decimal`(0.0425) 또는 `--rf-unit daily`를 지정하세요. 단위가 어긋나 보이면 경고를 출력합니다.
- 중복 날짜는 마지막 행만, 종가 결측 행은 제외, 무위험금리 결측은 직전 값으로 채웁니다.

## 2. 실행

```bash
pip install jumpmodels pandas numpy scikit-learn scipy matplotlib openpyxl
pip install hmmlearn      # HMM 벤치마크(--hmm)를 쓸 때만 필요

cd examples/rolling_refit
python run_pipeline.py --input 내데이터.xlsx --outdir out
```

형식 확인용 샘플 파일은 아래처럼 만들 수 있습니다(종가는 저장소의 나스닥 100 데이터, **무위험금리 열은 형식 예시를 위한 합성 데이터**입니다).

```bash
python make_sample_data.py --output sample_input.csv
python run_pipeline.py --input sample_input.csv --outdir out
```

구현의 핵심 규칙(변환의 인과성, 거래 지연, 거래비용, 재추정 스케줄, median filter)은 아래로 확인할 수 있습니다.

```bash
python test_pipeline.py      # 또는 pytest test_pipeline.py
```

파이썬에서 직접 호출할 수도 있습니다.

```python
from run_pipeline import run_pipeline

res = run_pipeline("내데이터.xlsx", outdir="out", jump_penalty=50.)
res["result"].regimes      # 온라인 추론 레짐
res["result"].params       # 재추정별 추정 파라미터
res["performance"]         # 전략 성과표
```

## 3. 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--feature-set` | `paper` | `paper`: 논문 Table 2의 3개 피처(DD hl=10, Sortino hl=20·60). `example`: 저장소 나스닥 예제의 9개 피처(hl 5·20·60 × 수익률·log DD·Sortino) |
| `--log-dd` | 꺼짐 | `paper` 피처의 downside deviation을 로그 변환 |
| `--warmup` | 252 | EWM 초기 불안정 구간으로 버릴 행 수 |
| `--jump-penalty` | 50.0 | 점프 페널티 λ. 클수록 레짐이 덜 바뀝니다(논문 Table 3) |
| `--window` | 3000 | 학습창 길이(거래일). 논문 기준 약 12년 |
| `--min-window` | 500 | 허용 최소 학습창. 데이터가 부족한 초기 재추정은 이 길이까지 자동 축소되고 경고를 출력 |
| `--refit-start` | 없음 | 이 날짜 이후의 재추정 시점만 사용(학습창이 짧은 초기 구간을 버리고 싶을 때) |
| `--delay` | 1 | 거래 지연. t일 신호는 t+delay+1일부터 적용(논문 §3.1) |
| `--cost-bps` | 10 | 편도 거래비용(bp) |
| `--start-date` / `--end-date` | 없음 | 분석 기간 필터 |
| `--n-init` | 10 | 좌표하강 재시작 횟수. 줄이면 빨라지고 해의 안정성은 떨어집니다 |
| `--delays` | `1,5,10` | 거래 지연 로버스트니스 표에 쓸 지연 일수 목록. `--no-robustness`로 생략 |
| `--no-plot`, `--save-features`, `--quiet` | | 출력 제어 |
| `--plot-font` | 자동 | 그림 폰트. 한글 라벨이 깨질 때 지정 (예: `NanumGothic`) |

### 3-1. 커스텀 변수 (`--extra-feature`, `--extra-file`)

수익률에서 파생된 기본 피처 외에 사용자 변수를 피처로 추가할 수 있습니다. 형식은 `열이름[:변환[:파라미터]]` 이고, 여러 번 지정하면 여러 피처가 됩니다.

| 변환 | 파라미터 | 계산 |
|---|---|---|
| `raw` (기본) | — | 원값 그대로 |
| `ewm` | 반감기(기본 20) | 지수가중이동평균 |
| `diff` | 시차(기본 1) | `x_t − x_{t−n}` |
| `pct` | 시차(기본 1) | `x_t / x_{t−n} − 1` |
| `log` | — | `log(x_t)` (양수 값만) |
| `logdiff` | 시차(기본 1) | `log(x_t) − log(x_{t−n})` (양수 값만) |

```bash
# 같은 파일 안의 '변동성지수' 열을 20일 반감기로 평활해 피처로 추가
python run_pipeline.py --input 내데이터.xlsx --extra-feature 변동성지수:ewm:20

# 여러 개 동시 지정
python run_pipeline.py --input 내데이터.xlsx \
    --extra-feature 변동성지수:ewm:20 --extra-feature 신용스프레드:diff:5

# 다른 파일(월간 매크로 등)에서 가져오기 — 날짜 기준 결합 후 직전 값으로 채움
python run_pipeline.py --input 내데이터.csv --extra-file 매크로.csv \
    --extra-feature 경기선행지수:diff:21
```

- `--extra-file`만 주고 `--extra-feature`를 생략하면 그 파일의 모든 열을 원값 그대로 피처로 씁니다.
- 모든 변환은 과거만 참조하므로 미래 정보가 새지 않습니다. 저빈도 변수는 직전 값으로 채워지며(look-ahead 없음), 값이 아직 없는 초기 구간은 피처 생성 단계에서 제외됩니다.
- 커스텀 변수도 학습창마다 클리핑·표준화가 다시 적합되므로 단위가 달라도 그대로 넣으면 됩니다.
- 파이썬에서는 `run_pipeline(..., extra_features=["VIX:ewm:20"], extra_file="매크로.csv")` 형태로 넘깁니다.

### 3-2. HMM 벤치마크 (`--hmm`)

논문 §3.3의 2-state 가우시안 HMM을 같은 기간·같은 0/1 전략으로 함께 돌려 비교합니다. `pip install hmmlearn`이 필요합니다.

```bash
python run_pipeline.py --input 내데이터.xlsx --hmm
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--hmm-window` | `--window`와 동일 | 학습·lookback 창 길이 |
| `--hmm-refit-every` | 21 | 파라미터 EM 재추정 주기(거래일). **1이면 논문과 동일한 매일 재추정** |
| `--hmm-smooth-k` | 6 | 온라인 추론 상태에 적용하는 median filter 길이 (Bulla et al. 2011) |
| `--hmm-n-init` | 10 | 초기값 개수(로그우도 최대 해 채택) |
| `--hmm-covariance-type` | `diag` | hmmlearn 공분산 형태 |
| `--hmm-simple-ret` | 꺼짐 | 로그수익률 대신 단순수익률로 적합 |

- 상태 구분은 논문과 같이 **조건부 변동성** 기준입니다(저변동성 = 0 = 위험자산 보유).
- 상태 추론(Viterbi)은 매일 수행되고, 재추정 주기만 옵션입니다. 3000일 창 EM 적합 1회가 약 0.8초라 매일 재추정하면 30년 데이터에서 몇 시간이 걸립니다. 기본값 21(월 1회)로 36년 데이터가 약 7분입니다.

## 4. 출력물 (`--outdir`)

| 파일 | 내용 |
|---|---|
| `regimes.csv` | 날짜별 온라인 추론 레짐(0=bull, 1=bear), 레짐 확률, 사용된 재추정 시점, 종가·수익률·무위험금리·초과수익률, 전략 비중과 전략 수익률 |
| `refit_params.csv` | 재추정 시점 × 레짐별 학습창 구간, 학습창 내 비중·연율 수익률·연율 변동성, 자기전이확률, **원래 피처 단위로 되돌린 군집 중심** |
| `strategy.csv` | 일별 비중·거래량·무위험금리·매수보유 수익률·전략 수익률 |
| `performance.csv` | 매수보유 vs JM 0/1 전략 성과(CAGR, 변동성, Sharpe, MDD, Calmar, ES 5%, Turnover, Leverage) |
| `insample_last_window.csv` | 마지막 학습창의 in-sample 레짐(온라인 추론과 비교용, 논문 Figure 4) |
| `regimes_cumret.png` | 누적 초과수익 곡선 + 현금 보유(bear) 구간 음영 (논문 Figure 5) |
| `refit_params.png` | 재추정에 따른 레짐별 군집 중심의 시간 변화 (논문 Figure 3, 연율 환산) |
| `weights.png` | 위험자산 비중 추이 |
| `delay_robustness.csv` | 거래 지연 1/5/10일 성과 비교 (논문 Table 5) |
| `hmm_regimes.csv` | `--hmm` 사용 시 HMM의 원(raw) 상태·평활된 레짐·사용된 재추정 시점 |
| `hmm_refit_params.csv` | HMM 재추정별 상태 조건부 연율 수익률·변동성·자기전이확률·로그우도 |
| `hmm_strategy.csv` | HMM 신호로 돌린 0/1 전략의 일별 결과 |

`--hmm`을 켜면 `performance.csv`와 `regimes_cumret.png`에 HMM 전략 열/곡선이 함께 들어갑니다.

## 5. 논문과의 대응 관계

| 논문 | 구현 위치 |
|---|---|
| §2 초과수익률 (지수 총수익 − 3개월 T-bill) | `data_io.load_market_data`: `excess_ret = ret − rf` |
| Table 2 피처(EWM DD hl=10, EWM Sortino hl=20·60) | `features.feature_engineer(ver="paper")` |
| 상태 정렬: 학습창 누적 초과수익이 높은 쪽이 bull(0) | `rolling.run_rolling_jm` → `JumpModel.fit(..., sort_by="cumret")` |
| §3.4.1 6개월마다 3000거래일 학습창으로 재추정 | `rolling.semiannual_anchors`, `rolling.refit_schedule` |
| §3.4.2 파라미터 고정 + 3000일 lookback 온라인 추론 | `rolling.run_rolling_jm` → `JumpModel.predict_proba_online` |
| §3.1 0/1 전략, 1일 지연, 편도 10bp | `backtest.run_0_1_strategy` |
| Table 4 성과 지표 | `backtest.performance_metrics` |
| Table 5 거래 지연 1/5/10일 로버스트니스 | `backtest.delay_robustness_table` |
| §3.3 HMM 벤치마크(3000일 롤링, Viterbi 온라인 추론, median filter) | `hmm_benchmark.run_rolling_hmm` |

## 6. 구현상의 선택과 주의사항

- **룩어헤드 없음**: 재추정 시점 `t`의 학습창은 `t` **직전** 거래일까지만 사용합니다. 클리핑(±3σ)과 표준화도 매 학습창에서 다시 적합해 이후 구간에 적용합니다.
- **온라인 추론 lookback**: 논문은 매일 길이 3000의 고정 lookback으로 DP를 풉니다. 여기서는 각 6개월 구간마다 `[재추정일 − window, 구간 끝]` 데이터를 `predict_proba_online`에 한 번 넣습니다. 구간 내 각 날짜의 lookback은 3000일 이상 3000+약 125일 이하가 되며, 어떤 날짜의 신호도 그 날짜 이후 정보를 쓰지 않습니다.
- **첫 영업일 기준**: 재추정 시점은 데이터에 실제로 존재하는 거래일 중 1월 1일·7월 1일 **이후 첫 거래일**입니다. 데이터 공백으로 45일 이상 떨어진 경우와 데이터의 첫 행은 제외합니다.
- **데이터가 짧을 때**: `--window`를 채우지 못하는 초기 재추정은 가용한 최대 길이(단, `--min-window` 이상)로 자동 축소하고 경고합니다. 학습창이 짧으면 군집이 불안정하므로, 초기 구간을 성과 평가에서 빼려면 `--refit-start`를 쓰세요.
- **λ 선택**: 논문 §3.4.3의 시계열 교차검증(매월, 8년 검증창, Sharpe 최대화)은 구현하지 않았습니다. `--jump-penalty`로 고정값을 주며, 논문의 대표값은 50.0입니다. HMM의 median filter 길이 `k`도 같은 이유로 고정값(기본 6)입니다.
- **HMM 재추정 주기**: 논문은 3000일 창을 매일 재적합하지만 EM 적합 1회가 약 0.8초라 30년 데이터에서 몇 시간이 걸립니다. 기본값은 21거래일(월 1회)이고 `--hmm-refit-every 1`로 논문과 동일하게 맞출 수 있습니다. 상태 디코딩(Viterbi)은 재추정 주기와 무관하게 매일 수행됩니다.
- **HMM 비교 구간**: HMM은 JM의 첫 온라인 추론일부터 시작하도록 맞춰 두 모델의 평가 구간이 같습니다. 다만 HMM은 워밍업으로 버린 수익률까지 학습에 쓸 수 있어 같은 시점에서 학습창이 더 길 수 있습니다(JM 피처는 `--warmup`만큼 늦게 시작).
- **지연 로버스트니스 표**: 모든 모델이 공통으로 덮는 기간으로 잘라 비교하며, 매수보유는 거래가 없어 지연과 무관하므로 한 열로만 표시합니다.
- **한글 라벨**: 커스텀 변수 이름이 한글이면 그림에 한글이 들어갑니다. 설치된 한글 지원 폰트를 자동으로 찾고, 없으면 경고합니다(`--plot-font`로 직접 지정 가능).
- **성과 지표 정의**: Return은 무위험수익을 포함한 CAGR, Sharpe는 연율 평균 초과수익÷연율 변동성, Calmar는 연율 평균 초과수익÷|MDD|, Turnover는 연 `Σ|Δw|/2`, ES는 일간 수익률 하위 5% 평균입니다. MDD는 총수익 복리 자산곡선 기준입니다.
- **거래 시작 시점**: 신호가 아직 없는 첫 `delay+1`일은 위험자산 100%로 둡니다.
- 그림은 LaTeX 의존성이 있는 `jumpmodels.plot` 대신 `plotting.py`에서 직접 그립니다.

## 7. 실행 예시 결과

`make_sample_data.py`로 만든 샘플(나스닥 100 종가 + 합성 무위험금리, 1989–2024, λ=50, 3000일 학습창, 1일 지연, 10bp)에 HMM 벤치마크를 함께 돌린 결과입니다.

```bash
python make_sample_data.py --output sample_input.csv
python run_pipeline.py --input sample_input.csv --hmm --outdir out
```

```
온라인 레짐 구간: 1989-01-03 ~ 2024-09-27 (9004거래일)
JM  bear 레짐 비중: 20.3%, 레짐 전환  31회 (연 0.87회)
HMM 고변동성 비중: 25.9%, 레짐 전환 126회 (연 3.53회)

             B & H  JM 0/1 HMM 0/1
Return       14.1%   13.9%   12.5%
Volatility   26.3%   19.7%   16.1%
Sharpe        0.53    0.62    0.64
MDD         -82.9%  -46.7%  -33.6%
Calmar        0.17    0.26    0.31
ES_0.05      -3.9%   -3.0%   -2.5%
Turnover      0.0%   44.8%  176.3%
Leverage    100.0%   79.7%   74.1%

거래 지연 로버스트니스 (지연 1, 5, 10일)
model   B & H     JM                  HMM
delay              1      5     10      1      5     10
Return  14.1%  13.9%  14.5%  13.2%  12.5%  13.2%  10.5%
Sharpe   0.53   0.62   0.64   0.58   0.64   0.67   0.52
Calmar   0.17   0.26   0.27   0.25   0.31   0.26   0.22
```

무위험금리가 합성 데이터라 논문의 S&P 500 결과와 직접 비교할 수는 없지만, 논문의 핵심 결론들이 그대로 재현됩니다.

- **지속성**: JM은 연 0.87회 전환(논문 Table 3의 λ=50~70 구간 0.5~0.8회), HMM은 연 3.53회(논문 k=8일 때 3.2회).
- **회전율**: JM 44.8% vs HMM 176.3% — 논문(S&P 500 기준 JM 44%, HMM 141%)과 같은 크기의 격차로, JM이 훨씬 적게 거래하고도 비슷한 위험 감축을 달성합니다.
- **거래 지연 내성**: JM의 Sharpe는 지연 1→10일에서 0.62→0.58로 완만하게 떨어지는 반면, HMM은 0.64→0.52로 더 빠르게 무너집니다(논문 Table 5와 같은 방향).
- 두 모델 모두 MDD와 ES를 크게 줄여 하방위험 감축이라는 전략의 목적을 달성합니다.

HMM 재추정 주기는 기본값 21거래일(월 1회)이며, 위 실행은 약 9분 걸렸습니다. 논문과 완전히 동일한 매일 재추정(`--hmm-refit-every 1`)은 같은 데이터에서 몇 시간이 걸립니다.
