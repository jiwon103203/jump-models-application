# 롤링 재추정 파이프라인 (Rolling Refit Pipeline)

csv/엑셀 파일 하나(날짜·종가·무위험금리)를 넣으면 논문 [Shu, Yu and Mulvey (2024), *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach*](https://doi.org/10.1057/s41260-024-00376-x)의 절차대로

1. 초과수익률 계산
2. 지수가중이동(EWM) downside deviation·Sortino ratio 피처 생성 (+ 원하면 **확장 피처 세트**나 **사용자 커스텀 변수** 추가)
3. **6개월마다(1월·7월 첫 영업일) 3000거래일 학습창으로 점프 모델 재추정** + 재추정 사이 구간은 온라인 추론 (sparse JM을 쓸 때는 **꼭 남기고 싶은 피처 고정** 가능)
4. 0/1 전략 백테스트(거래비용·거래지연 반영, **현금 비중 제한**과 **매수/매도 거래비용 분리** 지원) + **거래 지연 로버스트니스 표(논문 Table 5)**
5. **변수 유형별 가중 비중**(sjm)과 **bear 국면 분석 · 유사 국면 탐색 · 국면 종료 시나리오**
6. 선택적으로 **HMM 벤치마크(논문 §3.3)** 와 성과 비교

까지 수행하고 결과 csv와 그림을 저장합니다. 과거를 다시 걷지 않고 **지금 국면만** 알고 싶다면 `--inference`로 현재 반기 하나만 추론할 수 있습니다(3-11).

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
res["result"].regimes      # 온라인 추론 레짐 (regime + proba_0·proba_1 확률)
res["result"].params       # 재추정별 추정 파라미터
res["performance"]         # 전략 성과표
```

## 3. 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--feature-set` | `paper` | `paper`: 논문 Table 2의 3개 피처(DD hl=10, Sortino hl=20·60). `example`: 저장소 나스닥 예제의 9개 피처(hl 5·20·60 × 수익률·log DD·Sortino). `extra`: `example`의 수익률·Sortino 6개 + 수익률·변동성 파생 25개 + `DD_5`·`DD_10`·`DD_20`·`DD_60` = 35개 (3-1 참고) |
| `--log-dd` | 꺼짐 | `paper`·`extra`의 downside deviation을 로그 변환(`DD_10` → `DD-log_10`). `example`은 원래 로그 스케일이라 영향이 없습니다 |
| `--remove-series` | 없음 | 피처 **시리즈를 통째로 제거**. `--remove-series var` 는 `var_5`·`var_20`·`var_60`을 모두 뺍니다. 개별 피처 이름·커스텀 변수 지정도 받고, 여러 번 지정 가능 (3-7 참고) |
| `--warmup` | 252 | EWM 초기 불안정 구간으로 버릴 행 수 |
| `--model` | `jm` | `jm`: 논문의 원본 점프 모델 / `sjm`: 피처 선택이 있는 sparse 점프 모델 (3-5 참고) |
| `--no-cont` | 꺼짐(=연속형 사용) | 기본은 **연속형 점프 모델(CJM)** 이라 `regimes.csv`의 `proba_*`가 확률값입니다. 이 옵션을 주면 논문의 이산 모델로 돌아가 `proba_*`가 0/1 원핫이 됩니다 (3-8 참고) |
| `--grid-size` | 0.05 | 연속형 모델의 확률 격자 간격. 출력 확률의 해상도이자 계산량을 정합니다 (3-8 참고) |
| `--max-feats` | 피처 수의 절반 | `sjm` 전용. 남길 유효 피처 개수 κ² |
| `--pin-feature` | 없음 | `sjm` 전용. 재추정마다 **항상 남길 피처**. 피처 세트 이름·`custom`·`all`·개별 피처 이름·커스텀 변수 지정을 받고, 여러 번 지정 가능 (3-6 참고) |
| `--jump-penalty` | 50.0 | 점프 페널티 λ. 클수록 레짐이 덜 바뀝니다(논문 Table 3) |
| `--window` | 3000 | 학습창 길이(거래일). 논문 기준 약 12년 |
| `--min-window` | 500 | 허용 최소 학습창. 데이터가 부족한 초기 재추정은 이 길이까지 자동 축소되고 경고를 출력 |
| `--refit-start` | 없음 | 이 날짜 이후의 재추정 시점만 사용(학습창이 짧은 초기 구간을 버리고 싶을 때) |
| `--delay` | 1 | 거래 지연. t일 신호는 t+delay+1일부터 적용(논문 §3.1) |
| `--cost-bps` | 10 | 편도 거래비용(bp). 아래 매수/매도 옵션을 주지 않은 쪽에 적용 |
| `--buy-cost-bps` | `--cost-bps` | 매수 거래비용(bp) (3-3 참고) |
| `--sell-cost-bps` | `--cost-bps` | 매도 거래비용(bp). 증권거래세처럼 매도가 더 비쌀 때 지정 |
| `--max-cash` | 1.0 | bear 레짐에서 허용하는 최대 현금 비중(0~1) (3-3 참고) |
| `--min-cash` | 0.0 | bull 레짐에서도 항상 유지할 최소 현금 비중(0~1) |
| `--start-date` / `--end-date` | 없음 | 분석 기간 필터 |
| `--n-init` | 10 | 좌표하강 재시작 횟수. 줄이면 빨라지고 해의 안정성은 떨어집니다 |
| `--delays` | `1,5,10` | 거래 지연 로버스트니스 표에 쓸 지연 일수 목록. `--no-robustness`로 생략 |
| `--inference` | 꺼짐 | **추론 전용 모드**. 과거 롤링·백테스트를 건너뛰고 현재 반기 직전 `--window`거래일로 한 번 적합해 현재 반기만 추론 (3-11 참고) |
| `--weight-group` | `type` | `sjm` 전용. 피처 가중을 묶어 볼 기준 (`type` / `type-horizon` / `series` / `horizon`) (3-9 참고) |
| `--similar-target` | 마지막 국면 | 유사 국면 탐색의 기준 국면. 국면의 시작일이거나 국면 **안**의 날짜 (3-10 참고) |
| `--similar-metric` | 지표 전체 | 유사도에 쓸 지표를 직접 지정. 여러 번 지정 가능 |
| `--similar-top` | 3 | 시나리오 표에 남길 유사 국면 개수 |
| `--similar-include-later` | 꺼짐 | 기준 국면 **이후**에 일어난 국면도 비교 대상에 포함. 기본은 기준 국면 이전에 끝난 국면만 봅니다 (3-10 참고) |
| `--episode-horizon` | 22 | 국면 진입 후 전방 구간 길이(거래일). `ret_fwd`·`mdd_fwd`의 창 |
| `--false-signal-return` | 0.0 | 구간수익이 이 값을 넘으면 오탐으로 **표시**(제외는 하지 않음) |
| `--min-episode-len`, `--drop-false-signal` | 없음 / 꺼짐 | 유사도 후보와 길이 통계에서 제외할 국면. 둘 다 기본은 제외 없음 |
| `--path-horizon`, `--max-lag`, `--align-by` | 60, 30, `rmse` | 경로 비교 구간 길이, 탐색할 최대 시차, 정렬 기준(`rmse`/`corr`) |
| `--no-episodes` | 꺼짐 | 국면 분석 전체를 건너뜀 |
| `--episode-state` | 마지막 상태 | 국면을 끊을 상태 번호. `--n-components`를 3 이상으로 둘 때 씁니다 |
| `--no-plot`, `--save-features`, `--quiet` | | 출력 제어 |
| `--plot-font` | 자동 | 그림 폰트. 한글 라벨이 깨질 때 지정 (예: `NanumGothic`) |

### 3-1. 확장 피처 세트 (`--feature-set extra`)

`example`의 수익률·Sortino 6개에, 수익률과 변동성에서 흔히 쓰는 파생 변수 25개와 downside deviation 4개(`DD_5`·`DD_10`·`DD_20`·`DD_60`)를 더한 **35개 피처** 세트입니다. 수익률 시계열 하나만으로 만들 수 있는 통계를 폭넓게 깔아 두고 모델이 고르게 하는 용도이므로, **`--model sjm`과 함께 쓰는 것을 권장합니다.**

```bash
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm
```

`example`에서는 `ret_5/20/60`과 `sortino_5/20/60` **6개만** 그대로 가져오고, 로그 downside deviation(`DD-log_5/20/60`)은 가져오지 않습니다. 같은 통계를 로그·원 스케일 두 벌로 들고 있을 이유가 없어서, `extra`는 아래의 `DD_5/10/20/60` 한 벌만 씁니다(`--log-dd`를 주면 이 한 벌이 로그 스케일이 됩니다). 반감기 5·20·60일은 `example`이 쓰는 것과 같으므로, `--log-dd`를 주면 `example`의 로그 DD 3개가 그대로 `extra` 안에 들어옵니다.

| 세트 | downside deviation |
|---|---|
| `example` | `DD-log_5`, `DD-log_20`, `DD-log_60` (항상 로그) |
| `paper` | `DD_10` → `--log-dd` 시 `DD-log_10` |
| `extra` | `DD_5`, `DD_10`, `DD_20`, `DD_60` → `--log-dd` 시 `DD-log_5/10/20/60` |

`example`의 6개 외에 추가되는 29개는 아래와 같습니다.

| 피처 | 개수 | 계산 |
|---|---|---|
| `ret-simple` | 1 | 당일 (초과)수익률 `r_t` |
| `ret-log` | 1 | 로그수익률 `log(1 + r_t)` |
| `ret-abs` | 1 | 절대수익률 `\|r_t\|` |
| `ret-sq` | 1 | 제곱수익률 `r_t²` |
| `ret-cumlog` | 1 | 시작 시점부터의 누적 로그수익률 |
| `std_5/20/60` | 3 | 5·20·60일 롤링 표준편차 |
| `var_5/20/60` | 3 | 5·20·60일 롤링 분산 (= `std²`) |
| `mad_5/20/60` | 3 | 5·20·60일 롤링 평균절대수익률 |
| `rms_5/20/60` | 3 | 5·20·60일 롤링 RMS 수익률 |
| `vol-log_5/20/60` | 3 | 반감기 5·20·60일 EWMA 변동성(로그 스케일) |
| `vol-chg_5/20/60` | 3 | 위 변동성의 자기 horizon(5·20·60일) 대비 변화 |
| `vol-ratio_5-20`, `vol-ratio_20-60` | 2 | 단기/장기 변동성 비율 (로그 스케일이므로 차이가 곧 비율) |
| `DD_5`, `DD_10`, `DD_20`, `DD_60` | 4 | 반감기 5·10·20·60일 downside deviation. 기본은 원(raw) 스케일, `--log-dd`를 주면 `DD-log_5/10/20/60`. `DD_10`은 논문 Table 2의 피처, `DD_5/20/60`은 `example`이 쓰는 반감기입니다 |

- **Rolling Return은 넣지 않았습니다.** `example`의 `ret_5/20/60`(EWM 평균 수익률)이 이미 "과거 구간의 평균 수익률"이라 가중 방식만 다른 같은 통계이기 때문입니다. 단순 창(window) 방식의 누적수익률이 따로 필요하면 `features.extra_features`에 한 줄 추가하면 됩니다.
- EWMA 변동성(`vol-*`)은 **로그 스케일**을 씁니다. 양수·우측 꼬리 분포라 로그가 다루기 좋고, 로그의 차이가 곧 변화율·비율이 되어 `vol-chg`·`vol-ratio`가 자연스럽게 정의됩니다. downside deviation은 `--log-dd`로 정하며 기본은 원 스케일입니다.
- `DD` 계열이 하방 변동성만 보는 데 반해 `vol-log`는 양방향 변동성이라, 둘을 같이 쓰면 "하락"과 "변동성 확대"를 구분하는 데 도움이 됩니다.
- **downside deviation은 `--log-dd` 하나로 스케일이 정해집니다.** 기본은 원 스케일 `DD_5/10/20/60`, `--log-dd`를 주면 `DD-log_5/10/20/60`이고 열 개수는 35개로 같습니다. `log(DD_20) == DD-log_20`이라 두 스케일은 정보량이 같지만, 군집 거리(윈저라이징·표준화 후 유클리드)는 스케일에 따라 달라지므로 결과는 조금 달라집니다. `vol-log` 계열과 스케일을 맞추려면 `--log-dd`를 켜세요.
- **DD 반감기는 두 세트의 합집합입니다.** `DD_10`은 논문 Table 2의 대표 피처라 넣었고(그래서 `--pin-feature paper`가 `extra`에서 온전히 동작합니다), `DD_5/20/60`은 `example`이 쓰는 반감기라 그대로 가져왔습니다(`--log-dd`와 함께 쓰면 `--pin-feature example`도 온전히 동작합니다). `DD_5`는 `vol-log_5`·`rms_5`·`mad_5`가 보는 단기 변동성의 하방 전용 버전이라, 짧은 horizon에서 "하락"과 "변동성 확대"를 나눠 보는 데 쓰입니다. 반감기를 바꾸려면 `features.EXTRA_DD_HLS`를 고치고, 몇 개만 빼려면 `--remove-series DD_5`처럼 지정하세요(3-7).
- `std`와 `var`는 서로 제곱 관계라 정보가 같습니다. 요청하신 목록을 그대로 반영해 둘 다 넣었고, `sjm`을 쓰면 보통 한쪽만 남습니다. 아예 빼고 싶으면 `--remove-series var`를 쓰세요(3-7).
- ⚠️ `ret-cumlog`는 **정상(stationary) 시계열이 아닙니다.** 표본 전체에 걸쳐 추세를 그리므로 군집 좌표로는 적절하지 않고, 학습창 밖 구간에서는 윈저라이징(±`--clip-mul`σ) 경계에 붙어 상수처럼 동작합니다. 요청 목록에 있어 포함했지만 `sjm`은 대개 이 열의 가중을 0으로 떨어뜨립니다. `--remove-series ret-cumlog`로 아예 뺄 수 있습니다(3-7).
- 일반적으로 `ret-simple`·`ret-log`·`ret-sq`처럼 평활하지 않은 당일 값은 노이즈가 커서 `sjm`에서 먼저 탈락하고, 변동성 계열(`std`·`rms`·`vol-log`·`vol-ratio`)이 남는 경향이 있습니다. 어떤 변수가 실제로 선택됐는지는 `feat_weights.csv`·`feat_weights.png`와 실행 로그에서 확인하세요.
- 모든 통계는 과거만 참조합니다(`test_extra_feature_set_is_causal`에서 검증).
- 롤링 창 60일과 `vol-chg_60`의 60일 시차 때문에 앞부분이 잘려 나가므로, 기본 `--warmup 252`면 충분합니다.

### 3-2. 커스텀 변수 (`--extra-feature`, `--extra-file`)

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

### 3-3. 현금 비중 제한과 매수/매도 거래비용 (`--max-cash`, `--min-cash`, `--buy-cost-bps`, `--sell-cost-bps`)

논문의 0/1 전략은 bear 레짐에서 위험자산을 100% 팔고 bull 레짐에서 100% 담습니다. 실제 운용에서는 그렇게 극단적으로 움직이기 어렵고(위임 한도, 최소 투자비율), 매수와 매도의 비용도 대칭이 아닙니다. 두 옵션 묶음이 그 두 제약을 풀어 줍니다.

**현금 비중 제한.** `--max-cash`는 bear 레짐에서 현금(무위험자산)에 둘 수 있는 최대 비중, `--min-cash`는 bull 레짐에서도 항상 남겨 두는 최소 현금 비중입니다. 위험자산 비중은 `1 − max_cash`(bear)와 `1 − min_cash`(bull) 사이만 오갑니다.

```bash
# bear에서도 위험자산을 40%는 남기고(현금 최대 60%), bull에서도 현금 5%를 유지
python run_pipeline.py --input 내데이터.xlsx --max-cash 0.6 --min-cash 0.05
```

- 기본값 `--max-cash 1.0 --min-cash 0.0`이 논문 그대로의 0/1 전략입니다.
- 한 번의 레짐 전환에서 움직이는 폭이 `1.0` 대신 `max_cash − min_cash`이므로, 같은 거래비용 요율이라면 회전율(Turnover)과 연율 거래비용(`Cost`)도 그 비율만큼 줄어듭니다. 대신 bear 구간의 하방위험 감축 효과도 그만큼 약해집니다.
- `0 ≤ min_cash ≤ max_cash ≤ 1`을 어기면 실행 즉시 오류로 알려 줍니다.

**매수/매도 거래비용 분리.** `--buy-cost-bps`와 `--sell-cost-bps`는 비중 변화의 매수분과 매도분에 각각 다른 요율을 매깁니다. 지정하지 않은 쪽은 `--cost-bps`를 그대로 씁니다.

```bash
# 매수 10bp, 매도 25bp (한국 주식의 증권거래세를 감안한 예시)
python run_pipeline.py --input 내데이터.xlsx --buy-cost-bps 10 --sell-cost-bps 25
```

- 일별 비용은 `매수량 × buy_bps + 매도량 × sell_bps`로, `strategy.csv`의 `bought`·`sold`·`cost` 열에서 그대로 확인할 수 있습니다.
- 성과표의 `Cost` 행이 이렇게 차감된 연율 거래비용(전략 수익률에서 이미 빠진 값)입니다.
- 첫날은 거래로 보지 않습니다(초기 포지션이므로 비용 없음).

두 옵션은 함께 쓸 수 있고, 지연 로버스트니스 표(`delay_robustness.csv`)에도 같은 설정이 적용됩니다.

```bash
python run_pipeline.py --input 내데이터.xlsx \
    --max-cash 0.6 --min-cash 0.05 --buy-cost-bps 10 --sell-cost-bps 25
```

### 3-4. HMM 벤치마크 (`--hmm`)

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

### 3-5. Sparse JM (`--model sjm`)

논문이 쓰는 모델은 **원본(이산) JM** 입니다(`JumpModel(cont=False)`). 피처가 3개뿐이라 피처 선택의 실익이 없기 때문인데, 논문도 "확장된 피처 세트를 쓸 경우 sparse JM이 유용할 수 있다"고 언급합니다. `--feature-set extra`나 커스텀 변수로 피처를 늘렸다면 `--model sjm`으로 Nystrup et al. (2021)의 sparse JM을 쓸 수 있습니다.

```bash
# 예제 9피처 + 커스텀 2개 = 11피처를 sparse JM으로
python run_pipeline.py --input 내데이터.csv --model sjm --feature-set example \
    --extra-feature 변동성지수:ewm:20 --extra-feature 신용스프레드:diff:5
```

- 각 피처는 군집 분리력(BCSS)에 따라 가중되고, LASSO 형태의 제약 `‖w‖₁ ≤ κ²`이 노이즈 피처의 가중을 **0으로** 만듭니다. `--max-feats`가 κ²이며, 대략 "남길 유효 피처 개수"에 해당합니다(미지정 시 피처 개수의 절반).
- 가중은 **재추정마다 다시 계산**되므로, 어떤 변수가 시기별로 선택되는지 `feat_weights.csv`·`feat_weights.png`에서 확인할 수 있습니다. 실행 로그에도 재추정 평균 가중과 선택 비율이 요약됩니다.
- 점프 페널티는 라이브러리 내부에서 `1/√(피처 수)`로 나뉘므로 `--jump-penalty`는 JM과 비슷한 크기를 쓰면 됩니다.
- 피처가 3개 이하면 경고를 출력합니다. 그 경우엔 `--model jm`이 낫습니다.

### 3-6. 피처 고정 (`--pin-feature`)

`sjm`은 **재추정마다 피처를 다시 고릅니다.** 덕분에 노이즈 피처가 걸러지지만, 반대로 "논문 Table 2의 피처처럼 반드시 들어가야 하는 변수"가 어떤 시기에는 탈락합니다. `--pin-feature`로 고정한 피처는 **모든 재추정에서 가중이 0이 되지 않습니다.**

```bash
# paper 3개 피처(DD_10·sortino_20·sortino_60)를 고정하고 나머지 32개는 sjm이 알아서 고르게
# (사실상 "논문 피처 + 보조 피처")
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --pin-feature paper

# example 9개 전체 + 내가 넣은 VIX를 고정
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --extra-feature VIX:ewm:20 --pin-feature example --pin-feature VIX:ewm:20

# 특정 변수만 골라서 고정
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --pin-feature sortino_60 --pin-feature vol-log_20
```

`--pin-feature`에 줄 수 있는 값은 네 가지이고, 여러 번 지정하면 합집합이 됩니다.

| 지정 | 뜻 |
|---|---|
| `paper` / `example` / `extra` | 해당 피처 세트가 만드는 열 전체 (`--feature-set`과 같은 이름) |
| `custom` | `--extra-feature`·`--extra-file`로 넣은 커스텀 변수 전체 |
| `all` | 전체 피처 (= 피처 선택을 끄는 것과 같음) |
| `sortino_60`, `VIX_ewm20`, `VIX:ewm:20` | 개별 피처. 커스텀 변수는 `--extra-feature`에 준 지정을 그대로 써도 됩니다 |

동작 방식과 주의사항은 다음과 같습니다.

- **고정한 피처는 soft-threshold를 적용받지 않습니다.** 즉 자기 BCSS가 주는 가중을 그대로 쓰고, 나머지 피처만 `‖w‖₁ ≤ κ` 예산을 두고 경쟁합니다. 고정은 **"항상 들어간다"는 보장이지 "중요하게 쓰인다"는 보장이 아닙니다** — 레짐 구분력이 없는 피처를 고정하면 가중이 0은 아니되 아주 작아집니다.
- 원래 잘렸을 피처가 다시 들어오므로 **유효 피처 수가 `--max-feats`를 조금 넘길 수 있습니다.** 이게 고정의 비용입니다.
- `--max-feats`가 고정한 피처 개수보다 작으면 그 개수까지 자동으로 늘리고 경고합니다(고정한 피처는 어차피 모두 남으므로 그보다 줄일 수 없습니다). 미지정 시 기본값도 `max(2, 피처 수/2, 고정 개수)`가 됩니다.
- 고정한 피처가 하나도 없으면 계산은 기존 `SparseJumpModel`과 **완전히 동일**합니다(`test_solve_lasso_pinned`에서 검증).
- **`--feature-set`에 없는 열은 고정할 수 없습니다.** 예를 들어 `--feature-set example --pin-feature paper`는 두 세트가 공유하는 `sortino_20`·`sortino_60`만 고정하고, paper의 `DD_10`(반감기 10일)은 example 세트가 만들지 않으므로 경고와 함께 건너뜁니다. 세 피처를 모두 고정하려면 `--feature-set paper`나 `--feature-set extra`를 쓰세요 — `extra`는 `paper`의 세 열을 모두 포함하므로 `--pin-feature paper`가 온전히 적용되고, `--pin-feature DD_5`·`DD_10`·`DD_20`·`DD_60`처럼 개별 지정도 됩니다.
- **반대로 `--feature-set extra --pin-feature example`은 `--log-dd` 여부에 따라 갈립니다.** 기본(원 스케일)에서는 `extra`가 `DD-log_5/20/60`을 만들지 않으므로(3-1 참고) `ret_5/20/60`·`sortino_5/20/60` 6개만 고정하고 나머지는 경고와 함께 건너뜁니다. `--log-dd`를 같이 주면 `extra`의 DD 반감기(5·10·20·60)가 `example`의 5·20·60을 모두 덮으므로 `example` 9개가 전부 고정됩니다.
- **`--log-dd`는 고정할 이름도 바꿉니다.** `DD_10` → `DD-log_10`이 되므로 개별 지정도 그에 맞춰야 하고, `--pin-feature paper`처럼 그룹으로 지정하면 이름은 알아서 맞춰집니다.
- **`--remove-series`로 뺀 피처는 고정할 수 없습니다.** 두 옵션이 정면으로 모순되므로 경고가 아니라 오류입니다. 반대로 `--pin-feature extra`처럼 그룹으로 지정하면 남아 있는 열만 고정하고 제거한 열은 조용히 건너뜁니다(제거는 의도한 것이므로 경고하지 않습니다).
- `--model jm`에는 피처 선택 자체가 없으므로 지정해도 경고만 내고 무시합니다.
- 실제로 무엇이 고정됐는지는 실행 로그(`고정 피처: [...]`, 가중 요약의 `*` 표시)와 `feat_weights.png`(고정 피처는 실선)에서 확인할 수 있습니다.

### 3-7. 피처 시리즈 제거 (`--remove-series`)

`--feature-set`은 정해진 묶음이라 "extra는 쓰고 싶은데 이 계열만 빼고 싶다"는 조정이 안 됩니다. `--remove-series`가 그 자리로, **같은 통계를 horizon만 바꿔 만든 열 묶음(시리즈)을 통째로** 뺍니다.

```bash
# var_5·var_20·var_60을 빼고 extra 32개로 실행 (std가 이미 같은 정보를 담고 있으므로)
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --remove-series var

# 여러 번 지정하면 합집합 — 분산 3개 + 비정상 시계열 1개를 빼고 31개
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --remove-series var --remove-series ret-cumlog

# 시리즈 전체가 아니라 특정 열 하나만
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --remove-series DD_5
```

**시리즈 이름은 열 이름에서 `_` 앞부분**입니다. `var_20` → `var`, `vol-ratio_5-20` → `vol-ratio`, `DD_10` → `DD` 이고, `_`가 없는 `ret-cumlog`·`ret-simple` 같은 열은 그 자체가 하나의 시리즈입니다. `extra` 세트의 시리즈는 아래 15개입니다.

| 시리즈 | 제거되는 열 |
|---|---|
| `ret` | `ret_5`, `ret_20`, `ret_60` (EWM 평균 수익률) |
| `sortino` | `sortino_5`, `sortino_20`, `sortino_60` |
| `ret-simple` / `ret-log` / `ret-abs` / `ret-sq` / `ret-cumlog` | 같은 이름의 열 1개씩 |
| `std` / `var` / `mad` / `rms` | `*_5`, `*_20`, `*_60` |
| `vol-log` / `vol-chg` | `*_5`, `*_20`, `*_60` |
| `vol-ratio` | `vol-ratio_5-20`, `vol-ratio_20-60` |
| `DD` | `DD_5`, `DD_10`, `DD_20`, `DD_60` (`--log-dd` 시 `DD-log`) |

`--remove-series`에 줄 수 있는 값은 세 가지이고, 여러 번 지정하면 합집합이 됩니다.

| 지정 | 뜻 |
|---|---|
| `var`, `vol-chg`, `DD` | 시리즈 전체 |
| `var_20`, `ret-cumlog` | 개별 피처 한 개 |
| `VIX:ewm:20`, `VIX_ewm20` | 커스텀 변수. `--extra-feature`에 준 지정을 그대로 써도 되고, 변수 이름만 주면(`VIX`) 그 변수의 모든 변환이 빠집니다 |

주의사항은 다음과 같습니다.

- **`paper`·`example` 세트에도 그대로 적용됩니다.** 다만 열이 3~9개뿐이라 실익은 `extra`나 커스텀 변수를 많이 넣었을 때 큽니다.
- **`ret`와 `ret-simple`은 다른 시리즈입니다.** `--remove-series ret`은 EWM 평균 수익률 3개만 빼고 `ret-simple`·`ret-log`·`ret-abs`·`ret-sq`·`ret-cumlog`는 남깁니다. 앞글자가 같다고 함께 빠지지 않습니다.
- **아무 열도 맞지 않는 지정은 오류입니다.** 오타(`--remove-series vaar`)로 아무것도 제거되지 않은 채 실행이 끝나는 편보다, 사용 가능한 시리즈 목록과 함께 즉시 멈추는 편이 낫기 때문입니다(`--pin-feature`의 그룹 지정이 경고로 넘어가는 것과 다릅니다).
- **모든 피처를 제거하면 오류입니다.** 최소 한 개는 남아야 합니다.
- **`--log-dd`를 함께 쓰면 이름이 바뀝니다.** 시리즈 이름은 `DD` → `DD-log`, 개별 열은 `DD_5` → `DD-log_5`가 되므로 제거 지정도 그에 맞춰야 합니다.
- **제거는 행을 버리기 전에 일어납니다.** 즉 `vol-chg_60`(60일 시차)처럼 앞부분이 길게 NaN인 열을 빼면 그만큼 데이터가 되살아납니다. `--warmup`이 그보다 크면 차이는 없습니다.
- 무엇이 빠졌는지는 실행 로그의 `제거한 피처: [...]` 줄에서 확인할 수 있고, 제거한 열은 `feat_weights.csv`·`feat_weights.png`와 `refit_params.csv`에도 아예 나타나지 않습니다.

### 3-8. 연속형 점프 모델과 레짐 확률 (`--no-cont`, `--grid-size`)

`regimes.csv`의 `proba_0`·`proba_1`은 기본적으로 **확률값**입니다. 논문의 이산 점프 모델은 매일을 한 레짐에 딱 배정하기 때문에 두 열이 `1,0` 아니면 `0,1`인 원핫이 되고 `regime` 열을 되풀이할 뿐입니다. 그래서 이 저장소는 Nystrup, Lindström and Madsen (2020)의 **연속형 점프 모델(CJM)** 을 기본으로 씁니다. CJM은 동적계획법의 상태공간을 단체(simplex)의 꼭짓점이 아니라 그 위의 격자점으로 두기 때문에, 각 날짜의 해가 `proba_0 = 0.65` 같은 확률 벡터로 나옵니다.

```bash
# 기본 — proba_0·proba_1이 0.05 간격의 확률값
python run_pipeline.py --input 내데이터.xlsx

# 확률 해상도를 0.01로 (격자점이 늘어 온라인 추론이 느려집니다)
python run_pipeline.py --input 내데이터.xlsx --grid-size 0.01

# 논문 원본대로 이산 모델 — proba_0·proba_1이 0/1 원핫
python run_pipeline.py --input 내데이터.xlsx --no-cont
```

알아둘 점은 다음과 같습니다.

- **`regime` 열과 그 아래의 모든 결과는 성격이 바뀌지 않습니다.** `regime`은 여전히 확률이 가장 큰 상태(argmax)이고, 0/1 전략·성과표·그림은 그 신호를 그대로 씁니다. 다만 CJM과 이산 JM은 서로 다른 최적화 문제라 **추론된 레짐 경로 자체가 조금 달라지므로 백테스트 숫자도 달라집니다.** 논문 수치를 그대로 재현하려면 `--no-cont`를 쓰세요.
- **확률은 `--grid-size` 배수로만 나옵니다.** 기본 0.05면 `0, 0.05, ..., 1`의 21개 값입니다. 연속적인 실수가 필요하면 `--grid-size 0.01`처럼 줄이면 되지만, 격자점 수는 레짐 `n`개에 대해 `C(1/grid + n − 1, n − 1)`이고 동적계획법 비용은 그 **제곱**이라 금방 무거워집니다. 레짐 3개에 `--grid-size 0.01`은 5151개 격자점이라 실행 전에 오류로 막습니다(`--n-components 2`면 101개로 가볍습니다).
- **λ가 크면 대부분의 날은 여전히 0/1에 붙습니다.** CJM의 점프 페널티는 확률 벡터 사이 거리의 제곱에 비례해서, λ가 클수록 중간 확률을 쓰는 비용이 커집니다. 합성 데이터 예시에서 중간 확률(0 < `proba_0` < 1)이 나온 날의 비중은 λ=50에서 0.4%, λ=25에서 1.2%, λ=10에서 3.5%였습니다. 확률이 완만하게 움직이길 원한다면 `--jump-penalty`를 함께 낮춰야 합니다.
- **`--grid-size`의 역수는 정수여야 합니다.** `0.1`·`0.05`·`0.02`·`0.01`처럼 1을 나누는 값을 쓰세요. 아니면 가장 가까운 격자로 내림하면서 경고를 출력합니다.
- **`--model sjm`에도 그대로 적용됩니다.** sparse JM은 내부의 점프 모델에 설정을 그대로 넘기므로, 피처 가중·선택 결과와 무관하게 확률 출력이 됩니다.

### 3-9. 변수 유형별 가중치 (`--weight-group`)

`--model sjm`은 재추정마다 피처별 가중을 다시 정하고, 그 값이 `feat_weights.csv`·`feat_weights.png`에 남습니다. 그런데 `--feature-set extra`처럼 35개짜리 피처 세트를 쓰면 라인 35개가 겹친 그림이라 읽히지 않습니다. 궁금한 것도 보통 "`mad_20`이 어떻게 됐나"가 아니라 **"지금 이 모델은 어떤 종류의 변수로 국면을 가르고 있나"** 이고요.

`weight_groups.csv`·`weight_groups.png`가 그 질문에 답합니다. 피처를 유형으로 묶어 **재추정 시점별 가중 비중(합 100%)** 을 냅니다.

```bash
# 기본 — 변수 유형 4종 + 커스텀
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm

# "5일 실현변동성 vs 20일 실현변동성"처럼 기간까지 나눠 보기
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm     --weight-group type-horizon
```

묶는 기준은 4가지입니다.

| `--weight-group` | 그룹 | 예 |
|---|---|---|
| `type` (기본) | `return` / `realized-vol` / `log-vol` / `downside` / `custom` | `sortino_20` → `return` |
| `type-horizon` | 위 유형 × 기간 | `std_5` → `realized-vol_5` |
| `series` | 피처 시리즈(`features.feature_series_name`) | `std_5` → `std` |
| `horizon` | 기간만 | `std_5` → `5`, `ret-cumlog` → `none` |

`type`의 5개 그룹은 다음과 같이 나뉩니다(`weights.FEATURE_TYPE_SERIES`).

| 유형 | 포함 시리즈 | 뜻 |
|---|---|---|
| `return` | `ret`, `sortino`, `ret-simple`, `ret-log`, `ret-cumlog` | 수익률 수준과 그 위에 세운 위험조정 수익률 |
| `realized-vol` | `std`, `var`, `mad`, `rms`, `ret-abs`, `ret-sq` | 롤링 창에서 잰 양방향 변동성과 그 일간 프록시 |
| `log-vol` | `vol-log`, `vol-chg`, `vol-ratio` | 로그 스케일 EWMA 변동성, 그 변화와 기간 구조 |
| `downside` | `DD`, `DD-log` | 하방 편차 |
| `custom` | 그 외 전부 | `--extra-feature`로 넣은 변수 |

알아둘 점은 다음과 같습니다.

- **`--model sjm` 전용입니다.** 일반 JM은 피처 가중이라는 개념 자체가 없어(`feat_weights`가 `None`) 이 산출물이 나오지 않습니다.
- **비중은 L1 기준입니다.** sparse JM의 가중 `w`는 `‖w‖₂ = 1`로 정규화되어 있어서 그대로 더하면 100%가 되지 않습니다. 그래서 `w_j / Σ_j w_j`로 비중을 잡고, `‖w‖₁` 자체는 `total` 열에 따로 남깁니다.
- **`total`은 가중이 얼마나 퍼져 있는지를 봅니다.** 피처 하나에 다 몰리면 1에 가깝고, 여러 피처에 고르게 퍼질수록 커집니다(`√(선택된 피처 수)`가 상한).
- **모든 가중이 0인 재추정은 비중이 NaN입니다.** 학습창이 한 레짐만 덮었다는 뜻이고, 그 경우는 재추정 단계에서 이미 경고가 나옵니다.

### 3-10. 국면 분석과 유사 국면 탐색 (`--similar-*`, `--episode-*`)

레짐 시계열을 **bear 국면(연속된 구간) 단위로 끊어** 하나씩 성격을 재고, **지금 국면과 가장 닮은 과거 국면**을 찾아, 그 국면이 걸었던 길로 현재 국면의 종료 시점을 가늠합니다. 기본으로 켜져 있고 `--no-episodes`로 끕니다.

```bash
# 기본 — 마지막(현재) 국면을 기준으로
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm

# 특정 국면을 기준으로. 국면의 시작일이 아니라 국면 '안'의 날짜를 줘도 됩니다
python run_pipeline.py --input 내데이터.xlsx --similar-target 2020-04-15

# 오탐과 22일 미만 국면을 비교 대상에서 빼고, 지표도 직접 고르기
python run_pipeline.py --input 내데이터.xlsx --drop-false-signal --min-episode-len 22     --similar-metric vol_20_mean --similar-metric dd_at_entry --similar-metric mdd_fwd
```

**(1) 국면 추출과 측정** — `regime_episodes.csv`. 국면 하나가 한 행이고, 시작일이 인덱스입니다.

| 열 | 뜻 |
|---|---|
| `end` / `length` | 종료일과 길이(거래일) |
| `period_ret` / `mdd` | 그 구간의 벤치마크 누적수익과 MDD. 거래 지연을 넣지 않은 **시장 자체의 움직임**입니다(전략 성과는 `strategy.csv` 쪽) |
| `false_signal` | `period_ret`가 `--false-signal-return`(기본 0)을 넘으면 True. 방어로 갔는데 시장이 올랐다는 뜻 |
| `ongoing` | 데이터 마지막 날까지 이어지는 국면, 즉 현재 국면 |

**(2) 국면별 특성 지표** — 같은 파일의 뒷부분이며, 유사도 계산이 이 값들 위에서 돕니다.

| 열 | 뜻 |
|---|---|
| `loss_diff_mean` | 국면 평균 `loss(bear) − loss(bull)`. 음수가 클수록 피처가 bear 군집에 확실히 앉아 있었다는 뜻 |
| `distance_days` / `penalty_days` / `penalty_share` | 국면 길이를 **거리 요인**(피처가 실제로 bear 중심에 더 가까웠던 날)과 **페널티 요인**(bull 중심이 더 가까웠는데 점프 페널티가 붙잡아 둔 날)으로 나눈 것 |
| `vol_5_mean` / `vol_20_mean` | 국면 평균 5일·20일 실현변동성 (연율) |
| `sortino_20_mean` | 국면 평균 EWM Sortino (반감기 20) |
| `vol_ratio_state` | `vol_20_mean` ÷ 그 시점 재추정의 **bear 상태 학습창 변동성**. 그 모델의 기준으로 이 국면이 얌전했는지 사나웠는지 |
| `state_flip` | 그 재추정에서 bull 상태가 **저변동성** 쪽이면 1. 보통은 0이므로, 1은 상태 배치가 평소와 달랐던 재추정을 뜻합니다 |
| `dd_at_entry` | 국면 진입 시점에 **이미 실현되어 있던 낙폭**. 얼마나 선제적인 신호였는지 |
| `ret_fwd` / `mdd_fwd` / `fwd_days` | 진입 후 `--episode-horizon`(기본 22)거래일의 누적수익·MDD와, 데이터에 실제로 있었던 날 수 |

**(3) 유사 국면 순위** — `similar_episodes.csv`. 비교 대상은 **기준 국면이 시작하기 전에 이미 끝난 국면**뿐입니다. 아직 일어나지도 않은 국면은 "이 국면이 얼마나 갈까"의 근거가 될 수 없기 때문이고, 기준 국면이 마지막(현재) 국면이면 어차피 모든 후보가 과거라 차이가 없습니다. 순수하게 "어떤 국면들이 서로 닮았나"만 보고 싶으면 `--similar-include-later`로 이 제한을 풉니다. 그 위에서, 지표를 국면 간에 표준화(z-score)한 뒤 **표준화된 차이의 RMS**를 거리로 씁니다. 합이 아니라 평균이라, 한 국면에만 결측인 지표가 있어도 그 국면만 그 지표를 빼고 계산합니다(`n_metrics`가 실제로 쓴 지표 수). 국면 간에 값이 전혀 변하지 않는 지표는 정보가 없으므로 자동으로 빠집니다.

**(4) 경로 비교** — `similar_episode_paths.csv`·`.png`. 기준 국면과 1순위 국면의 종가를 각자 **진입일 = 1.0**으로 정규화해 나란히 놓고, 둘을 가장 잘 겹치는 **시차**를 `[−max-lag, +max-lag]`에서 찾습니다. 시차가 `+18`이면 **기준 국면이 18일 앞서 간다**는 뜻입니다(같은 지점에 18일 먼저 도달). 변동성이 더 큰 국면이 같은 낙폭에 더 빨리 닿기 때문에 생기는 차이입니다. 기준은 `--align-by rmse`(수준 차이 최소화, 기본)와 `corr`(상관 최대화) 중에 고르며, `rmse` 쪽이 더 엄격합니다.

**(5) 종료 시점 시나리오** — `episode_length_scenarios.csv`. (3)과 같은 후보 집합에서 잰 과거 국면 길이의 분위수(`p25`/`p50`/`p75`)와 유사 국면 상위 `--similar-top`개의 길이를, 각각 현재 국면의 총 길이로 놓고 잔여일과 예상 종료일을 냅니다.

알아둘 점은 다음과 같습니다.

- **오탐·최소 길이 필터는 기본적으로 켜지지 않습니다.** 수익이 났던 국면이나 3일짜리 국면을 비교 대상에 넣을지는 데이터가 아니라 시장에 대한 판단이라, 표에는 `false_signal`과 `length`를 그대로 남기고 제외는 `--drop-false-signal`·`--min-episode-len`으로 **명시적으로** 요청할 때만 합니다. `--episode-horizon`과 같은 값(기본 22)을 `--min-episode-len`에 주면 전방 구간이 국면 안에서 온전히 관측된 국면만 남습니다.
- **특성 지표는 사후 서술입니다.** `ret_fwd`·`mdd_fwd`는 진입일 이후의 데이터를 봅니다. 모델 자체에는 룩어헤드가 없지만(6절), 이 표는 **지나간 국면을 설명하는 자료**이지 진입 시점에 쓸 수 있는 신호가 아닙니다.
- **`loss_*` 없이도 돕니다.** 이 기능이 생기기 전에 만든 `regimes.csv`에는 `loss_0`·`loss_1`이 없습니다. 그러면 손실 기반 3개 열만 NaN이 되고 나머지 지표로 유사도를 계산합니다.
- **예상 종료일의 근사.** 데이터 마지막 날을 넘어가는 예측은 남은 일수를 영업일로 셉니다. 휴장일을 모르니 긴 예측일수록 조금 이르게 찍힙니다.
- **잔여일이 음수일 수 있습니다.** 기준 국면이 이미 그 시나리오보다 오래 갔다는 뜻이고, 감추기보다 그대로 보여줍니다.
- **국면이 하나뿐이거나 경로가 겹치지 않으면** 해당 산출물만 경고와 함께 건너뛰고 나머지는 그대로 저장합니다. 현재 국면이 데이터 끝에 막 시작된 경우(경로가 10일도 안 됨)가 대표적입니다.

### 3-11. 추론 전용 모드 (`--inference`)

기본 실행은 **백테스트**입니다. 1990년부터 6개월마다 재추정하며 모든 날의 레짐을 추론하죠. 모델을 평가할 때는 그게 맞지만, **"지금 국면이 뭔가"** 만 알고 싶을 때는 과거를 전부 다시 걸을 이유가 없습니다.

`--inference`는 그 걸음의 **마지막 한 칸만** 수행합니다. 현재 반기 직전 `--window`(기본 3000)거래일로 **한 번 적합**하고, 그 반기의 첫 거래일부터 데이터 마지막 날까지를 온라인 추론합니다.

```bash
# 현재 국면만
python run_pipeline.py --input 내데이터.xlsx --inference

# 백테스트와 같은 설정으로
python run_pipeline.py --input 내데이터.xlsx --feature-set extra --model sjm --inference
```

```
추론 모드 — 현재 반기만 (재추정 2026-07-01)
학습창: 2014-07-02 ~ 2026-06-30 (3000거래일)
추론 구간: 2026-07-01 ~ 2026-09-04 (46거래일, bear 비중 47.8%)
------------------------------------------------------------------------
현재 국면(2026-09-04): BEAR (확률 100%) — 22거래일째 (2026-08-05부터)
권장 위험자산 비중: 0% (거래 지연 1일 반영, 현금 0%~100%)
변수 type별 가중 비중: realized-vol 59%, log-vol 22%, downside 19%
```

**현재 반기**는 데이터가 덮는 가장 최근 반기 앵커 — 1월 1일·7월 1일 **이후 첫 거래일** — 부터입니다(3절의 재추정 스케줄과 같은 규칙). 오늘이 9월 5일이면 7월 1일에 적합된 모델로 7월 이후를 추론합니다.

**백테스트와 완전히 같은 신호가 나옵니다.** 두 모드가 겹치는 날짜의 `regime`·`proba_*`·`loss_*`는 부동소수점 수준까지 동일합니다. 적합은 반기 이전 창만 보고, 반기 안 어떤 날의 추론도 그 창 + 그 날까지의 데이터만 보기 때문입니다(6절 룩어헤드 규칙 그대로). 차이는 **얼마나 계산하느냐**뿐입니다 — 재추정 30회 대신 1회라, 예제 데이터에서 20분대 → 6초였습니다.

산출물은 전부 `inference_` 접두사가 붙어서, 백테스트 결과와 **같은 폴더를 써도 덮어쓰지 않습니다**.

| 파일 | 내용 |
|---|---|
| `inference_regimes.csv` | 현재 반기의 날짜별 레짐·확률·상태 손실 + 종가·수익률·무위험금리·초과수익률과 **권장 위험자산 비중** |
| `inference_summary.csv` | 한 눈에 보는 요약: 기준일, 재추정일, 학습창 구간·길이, 추론 구간, 현재 레짐과 확률, 현재 국면 지속일수와 시작일, 반기 내 bear 비중, 권장 비중, 종가 |
| `inference_refit_params.csv` | 그 한 번의 재추정에 대한 레짐별 파라미터 (일반 `refit_params.csv`와 같은 형식) |
| `inference_feat_weights.csv` · `inference_weight_groups.csv` | `--model sjm` 사용 시 그 재추정의 피처 가중과 유형별 비중 (3-9) |
| `inference_regimes.png` | 반기 종가 + bear 구간 음영, 아래에 P(bear) 패널. 기준일은 점으로 표시 |

알아둘 점은 다음과 같습니다.

- **백테스트 전용 옵션은 쓰이지 않습니다.** 거래비용(`--cost-bps` 계열), `--delays`/`--no-robustness`, `--hmm`, 국면 분석(`--episode-*`·`--similar-*`), `--refit-start`는 이 모드에 대응물이 없습니다. `--hmm`을 주면 무시한다고 경고합니다. `--delay`와 `--min-cash`/`--max-cash`는 **권장 비중을 계산할 때만** 쓰입니다.
- **권장 비중은 성과가 아니라 신호를 다시 쓴 것입니다.** 레짐을 현금 한도로 매핑하고 거래 지연을 반영한 값이라, 0/1 전략이 지금 들고 있을 포지션과 같습니다. 백테스트를 돌리지 않았으므로 이 모드에는 성과표가 없습니다.
- **데이터가 짧으면 실패합니다.** 최근 반기 앵커 앞에 `--min-window`(기본 500)거래일이 없으면 재추정 시점을 만들 수 없다는 오류가 납니다. 백테스트와 같은 조건입니다.
- **반기가 막 시작했으면 추론 구간이 며칠뿐입니다.** 예컨대 7월 3일에 돌리면 1~2거래일이고, 그래도 동작합니다. 다만 `--jump-penalty`가 상태 전환을 억제하므로 반기 초반의 신호는 학습창 끝의 상태에 끌리는 경향이 있습니다.

## 4. 출력물 (`--outdir`)

아래는 기본(백테스트) 실행의 산출물입니다. `--inference`는 `inference_` 접두사가 붙은 별도 파일을 냅니다(3-11).

| 파일 | 내용 |
|---|---|
| `regimes.csv` | 날짜별 온라인 추론 레짐(0=bull, 1=bear), 레짐 확률(`proba_0`·`proba_1`), 상태별 손실(`loss_0`·`loss_1`), 사용된 재추정 시점, 종가·수익률·무위험금리·초과수익률, 전략 비중과 전략 수익률. 기본(연속형)에서는 `proba_*`가 `--grid-size` 간격의 확률값이고, `--no-cont`를 주면 0/1 원핫입니다 (3-8 참고) |
| `refit_params.csv` | 재추정 시점 × 레짐별 학습창 구간, 학습창 내 비중·연율 수익률·연율 변동성, 자기전이확률, **원래 피처 단위로 되돌린 군집 중심** |
| `strategy.csv` | 일별 비중·매수량(`bought`)·매도량(`sold`)·총 거래량(`traded`)·거래비용(`cost`)·무위험금리·매수보유 수익률·전략 수익률 |
| `performance.csv` | 매수보유 vs JM 0/1 전략 성과(CAGR, 변동성, Sharpe, MDD, Calmar, ES 5%, Turnover, Leverage, 연율 거래비용 `Cost`) |
| `insample_last_window.csv` | 마지막 학습창의 in-sample 레짐(온라인 추론과 비교용, 논문 Figure 4) |
| `regimes_cumret.png` | 누적 초과수익 곡선 + 방어적 비중을 든 bear 구간 음영 (논문 Figure 5). 현금 비중 제한을 걸면 범례가 그때의 위험자산 비중을 표시 |
| `refit_params.png` | 재추정에 따른 레짐별 군집 중심의 시간 변화 (논문 Figure 3, 연율 환산) |
| `weights.png` | 위험자산 비중 추이 (현금 비중 제한을 걸면 `1 − max_cash` ~ `1 − min_cash` 사이에서 움직임) |
| `delay_robustness.csv` | 거래 지연 1/5/10일 성과 비교 (논문 Table 5) |
| `feat_weights.csv` | `--model sjm` 사용 시 재추정별 피처 가중(0이면 그 시점에 탈락) |
| `feat_weights.png` | 재추정에 따른 피처 가중의 시간 변화. 고정한 피처는 실선, 나머지는 점선 |
| `weight_groups.csv` | `--model sjm` 사용 시 재추정별 **변수 유형별 가중 비중**(합 100%)과 `‖w‖₁`(`total` 열) (3-9) |
| `weight_groups.png` | 같은 내용의 100% 스택 그림 |
| `regime_episodes.csv` | bear 국면별 길이·구간수익·MDD·오탐 표시와 특성 지표(거리/페널티 분해 포함) (3-10) |
| `episode_lengths.png` | 국면 길이를 거리 요인 / 페널티 요인으로 나눈 스택 바 |
| `similar_episodes.csv` | 기준 국면과의 표준화 거리 순위 |
| `episode_length_scenarios.csv` | 과거 길이 분위수·유사 국면 길이별 잔여일과 예상 종료일 |
| `similar_episode_paths.csv` | 기준 국면과 1순위 국면의 정규화 경로(진입일 = 1.0) |
| `similar_episode_paths.png` | 같은 경로의 원본 / 시차 정렬 2패널 비교 |
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
| §3.1 0/1 전략, 1일 지연, 편도 10bp | `backtest.run_0_1_strategy` (기본값이 논문 설정. 현금 비중 제한·매수/매도 비용 분리는 논문 밖의 확장) |
| Table 4 성과 지표 | `backtest.performance_metrics` |
| Table 5 거래 지연 1/5/10일 로버스트니스 | `backtest.delay_robustness_table` |
| §3.3 HMM 벤치마크(3000일 롤링, Viterbi 온라인 추론, median filter) | `hmm_benchmark.run_rolling_hmm` |
| §3.4.1 각주의 sparse JM (Nystrup et al. 2021) | `rolling.init_model(model="sjm")` → `SparseJumpModel` |
| (논문 밖) 연속형 JM (Nystrup et al. 2020) — 확률로 나오는 레짐 | `rolling.init_model(cont=True)` → `JumpModel(cont=True)` |
| (논문 밖) 특정 피처를 선택에서 제외하고 항상 유지 | `features.resolve_pinned_features` → `sparse_pin.PinnedSparseJumpModel` |
| (논문 밖) 피처 시리즈를 모델에 넣기 전에 제거 | `features.resolve_removed_features` → `features.build_features(remove_series=...)` |
| (논문 밖) 변수 유형별 가중 비중 | `weights.group_feature_weights` → `plotting.plot_weight_groups` |
| (논문 밖) 국면 길이의 거리 요인 / 페널티 요인 분해 | `rolling.state_losses` → `regime_episodes.episode_metrics` |
| (논문 밖) 현재 반기만 추론하는 실행 모드 | `rolling.run_rolling_jm(last_refit_only=True)` → `run_pipeline.run_inference` |
| (논문 밖) 유사 국면 탐색과 국면 종료 시점 | `regime_episodes.rank_similar_episodes`·`compare_episode_paths`·`length_scenarios` |

## 6. 구현상의 선택과 주의사항

- **룩어헤드 없음**: 재추정 시점 `t`의 학습창은 `t` **직전** 거래일까지만 사용합니다. 클리핑(±3σ)과 표준화도 매 학습창에서 다시 적합해 이후 구간에 적용합니다.
- **온라인 추론 lookback**: 논문은 매일 길이 3000의 고정 lookback으로 DP를 풉니다. 여기서는 각 6개월 구간마다 `[재추정일 − window, 구간 끝]` 데이터를 `predict_proba_online`에 한 번 넣습니다. 구간 내 각 날짜의 lookback은 3000일 이상 3000+약 125일 이하가 되며, 어떤 날짜의 신호도 그 날짜 이후 정보를 쓰지 않습니다.
- **첫 영업일 기준**: 재추정 시점은 데이터에 실제로 존재하는 거래일 중 1월 1일·7월 1일 **이후 첫 거래일**입니다. 데이터 공백으로 45일 이상 떨어진 경우와 데이터의 첫 행은 제외합니다.
- **데이터가 짧을 때**: `--window`를 채우지 못하는 초기 재추정은 가용한 최대 길이(단, `--min-window` 이상)로 자동 축소하고 경고합니다. 학습창이 짧으면 군집이 불안정하므로, 초기 구간을 성과 평가에서 빼려면 `--refit-start`를 쓰세요.
- **연속형 모델이 기본**: `regimes.csv`에 레짐 확률을 남기기 위해 기본값을 연속형 JM(CJM)으로 두었습니다. 논문 수치를 재현할 때는 `--no-cont`로 이산 모델을 쓰세요(3-8 참고). 어느 쪽이든 `regime` 열은 확률이 가장 큰 상태이고, 확률이 정확히 반씩 갈리는 날은 낮은 번호(bull) 쪽으로 배정됩니다.
- **λ 선택**: 논문 §3.4.3의 시계열 교차검증(매월, 8년 검증창, Sharpe 최대화)은 구현하지 않았습니다. `--jump-penalty`로 고정값을 주며, 논문의 대표값은 50.0입니다. HMM의 median filter 길이 `k`도 같은 이유로 고정값(기본 6)입니다.
- **HMM 재추정 주기**: 논문은 3000일 창을 매일 재적합하지만 EM 적합 1회가 약 0.8초라 30년 데이터에서 몇 시간이 걸립니다. 기본값은 21거래일(월 1회)이고 `--hmm-refit-every 1`로 논문과 동일하게 맞출 수 있습니다. 상태 디코딩(Viterbi)은 재추정 주기와 무관하게 매일 수행됩니다.
- **HMM 비교 구간**: HMM은 JM의 첫 온라인 추론일부터 시작하도록 맞춰 두 모델의 평가 구간이 같습니다. 다만 HMM은 워밍업으로 버린 수익률까지 학습에 쓸 수 있어 같은 시점에서 학습창이 더 길 수 있습니다(JM 피처는 `--warmup`만큼 늦게 시작).
- **지연 로버스트니스 표**: 모든 모델이 공통으로 덮는 기간으로 잘라 비교하며, 매수보유는 거래가 없어 지연과 무관하므로 한 열로만 표시합니다.
- **Sparse JM의 중심값**: 라이브러리의 `SparseJumpModel.centers_`는 가중된 공간의 값이라 그대로는 해석하기 어렵습니다. `refit_params.csv`에는 가중하지 않은 피처 공간에서 다시 계산한 뒤 표준화를 되돌린 값을 기록합니다. 자기전이확률은 내부 `jm_ins.transmat_`에서 가져옵니다.
- **한 레짐만 나타나는 학습창**: 학습창이 한 레짐만 덮으면 해당 상태의 파라미터가 NaN이 되고 경고를 출력합니다. 학습창이 너무 짧거나 λ가 너무 클 때 발생합니다.
- **한글 라벨**: 커스텀 변수 이름이 한글이면 그림에 한글이 들어갑니다. 설치된 한글 지원 폰트를 자동으로 찾고, 없으면 경고합니다(`--plot-font`로 직접 지정 가능).
- **성과 지표 정의**: Return은 무위험수익을 포함한 CAGR, Sharpe는 연율 평균 초과수익÷연율 변동성, Calmar는 연율 평균 초과수익÷|MDD|, Turnover는 연 `Σ|Δw|/2`, ES는 일간 수익률 하위 5% 평균입니다. MDD는 총수익 복리 자산곡선 기준입니다.
- **거래 시작 시점**: 신호가 아직 없는 첫 `delay+1`일은 위험자산 100%로 둡니다.
- **국면 특성 지표는 사후 서술입니다.** `regime_episodes.csv`의 전방 구간 열(`ret_fwd`·`mdd_fwd`)은 진입일 **이후**의 데이터를 씁니다. 모델 신호에는 룩어헤드가 없지만 이 표는 지나간 국면을 설명하는 자료이지 실시간 신호가 아닙니다(3-10).
- **상태별 손실(`loss_*`)은 모델 자신의 손실입니다.** 가중된 피처 공간에서 `½‖x·w − μ_k‖²`이며(`jumpmodels.jump.do_E_step`와 같은 식), 점프 페널티와 맞바꿔지는 항이 바로 이 값입니다. 그래서 "가장 가까운 중심 ≠ 배정된 레짐"인 날이 페널티가 붙잡고 있는 날이 됩니다.
- 그림은 LaTeX 의존성이 있는 `jumpmodels.plot` 대신 `plotting.py`에서 직접 그립니다.

## 7. 실행 예시 결과

`make_sample_data.py`로 만든 샘플(나스닥 100 종가 + 합성 무위험금리, 1989–2024, λ=50, 3000일 학습창, 1일 지연, 10bp)에 HMM 벤치마크를 함께 돌린 결과입니다. 아래 7절의 숫자는 모두 **논문의 이산 모델**(`--no-cont`)로 얻은 것입니다. 기본값인 연속형 모델(3-8)은 레짐 경로가 조금 달라 성과 숫자도 그대로 재현되지는 않습니다.

```bash
python make_sample_data.py --output sample_input.csv
python run_pipeline.py --input sample_input.csv --hmm --no-cont --outdir out
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

### 7-1. 피처를 늘렸을 때 JM vs Sparse JM

같은 샘플에 예제 9피처 + 커스텀 2개(`변동성지수:ewm:20`, `변동성지수:diff:5`) = 11피처를 넣고, 2009–2024 구간을 비교한 결과입니다(`--n-init 3`, `--no-cont`).

| | JM | SJM (κ²=5.5, 기본) | SJM (κ²=3) |
|---|---|---|---|
| Sharpe | 0.75 | 0.61 | 0.55 |
| MDD | −21.2% | −25.3% | −27.8% |
| Turnover | 98.5% | 62.4% | 52.5% |
| 레짐 전환 | 연 1.97회 | 연 1.25회 | 연 0.99회 |

SJM은 회전율을 크게 낮추고 레짐을 더 지속적으로 만들지만, 이 표본에서는 위험조정수익이 JM보다 낫지는 않았습니다. κ²를 줄일수록 신호가 더 매끄러워지는 방향이 뚜렷하니, 두 모델과 κ² 값은 **본인 데이터에서 직접 비교해 보고 고르는 것을 권합니다**.

선택된 피처 가중(31회 재추정 평균, κ²=5.5)은 아래와 같습니다. 논문·라이브러리 예제와 같은 패턴 — 반감기가 긴 위험 피처가 높은 가중을 받고, 5일 반감기의 노이즈가 큰 수익률 피처는 대부분 탈락 — 이 나타나며, 추가한 커스텀 변수가 가장 높은 가중을 받았습니다.

| 피처 | 평균 가중 | 선택된 비율 |
|---|---|---|
| 변동성지수_ewm20 (커스텀) | 0.739 | 100% |
| DD-log_20 | 0.730 | 100% |
| DD-log_60 | 0.683 | 100% |
| DD-log_5 | 0.538 | 100% |
| sortino_60 | 0.476 | 100% |
| ret_60 | 0.421 | 100% |
| sortino_20 | 0.163 | 74% |
| ret_20 | 0.123 | 77% |
| sortino_5 | 0.034 | 26% |
| ret_5 | 0.007 | 19% |
| 변동성지수_diff5 (커스텀) | 0.005 | 13% |

논문 Table 2의 피처인 `sortino_20`도 재추정의 26%에서 탈락하는데, 이렇게 "꼭 남기고 싶은 피처가 시기마다 빠지는" 상황이 `--pin-feature`(3-6)를 쓰는 자리입니다. `--pin-feature paper`를 붙이면 이 열들의 선택 비율이 전부 100%가 됩니다.
