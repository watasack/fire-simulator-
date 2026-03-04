"""
年金受給開始年齢の最適化モジュール

FIRE達成時期と年金受給開始年齢を同時に最適化する。

定式化:
    minimize   m*  (FIRE達成月)
    subject to baseline_final_assets(m*, a_修平, a_桜) >= safety_margin
    where      a_修平, a_桜 ∈ {62, 63, ..., 75}

解法:
    Phase 1: FIRE前シミュレーション（FIRE判定なし、全月の状態記録）
    Phase 2: 確定的スクリーニング（全候補を高速評価、安全マージン以上で最小FIRE月を選択）
"""

import copy
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

_PROJECT_ROOT = str(Path(__file__).parent.parent)

from src.simulator import simulate_future_assets


PENSION_AGE_MIN = 62
PENSION_AGE_MAX = 75
DEFAULT_MIN_BASELINE_FINAL_ASSETS = 1_000_000
_REFINEMENT_TOP_K = 30

_CASH_STRATEGY_COLS = ['cash_target_reserve', 'cash_crash_threshold']


def _apply_cash_strategy(config: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    """現金管理戦略のオーバーライドを適用した config のコピーを返す。"""
    cfg = copy.copy(config)
    base = config['post_fire_cash_strategy']
    cfg['post_fire_cash_strategy'] = {**base, **strategy}
    return cfg


def optimize_pension_start_ages(
    current_cash: float,
    current_stocks: float,
    config: Dict[str, Any],
    scenario: str = 'standard',
    monthly_income: float = 0,
    monthly_expense: float = 0,
    min_baseline_final_assets: float = DEFAULT_MIN_BASELINE_FINAL_ASSETS,
    fire_month_search_range: int = 36,
    fire_month_step: int = 12,
    extra_budget_candidates: List[float] = None,
    cash_strategy_candidates: List[Dict[str, Any]] = None,
    pre_fire_investment_candidates: List[Dict[str, Any]] = None,
    reduction_candidates: List[float] = None,
    austerity_months: int = 120,
    min_asset_floor: float = 0,
) -> Dict[str, Any]:
    """
    FIRE達成時期・年金受給開始年齢・FIRE後追加予算・現金管理戦略を同時に最適化する。

    決定論的ベースラインの最終資産が安全マージン以上となる最も早いFIRE月を求める。

    Args:
        current_cash: 現在の現金残高
        current_stocks: 現在の株式残高
        config: 設定辞書
        scenario: シナリオ名
        monthly_income: 月次労働収入
        monthly_expense: 月次支出
        min_baseline_final_assets: ベースライン最終資産の安全マージン（デフォルト: 300万円）
        fire_month_search_range: 基準FIRE月 ± この月数を探索
        fire_month_step: FIRE候補月の刻み幅
        extra_budget_candidates: FIRE後追加月額予算の候補リスト（円）
        cash_strategy_candidates: 現金管理戦略の候補リスト

    Returns:
        最適化結果の辞書
    """
    if extra_budget_candidates is None:
        extra_budget_candidates = [0]
    if cash_strategy_candidates is None:
        cash_strategy_candidates = [{}]
    if pre_fire_investment_candidates is None:
        pre_fire_investment_candidates = [{}]
    if reduction_candidates is None:
        reduction_candidates = [0.0]
    print("=" * 60)
    print("年金受給開始年齢の最適化")
    print("=" * 60)

    people = config['pension']['people']
    person_names = [p['name'] for p in people]
    start_age = config['simulation']['start_age']

    # Phase 0: FIRE前投資戦略スクリーニング
    if len(pre_fire_investment_candidates) > 1:
        print(f"\nPhase 0: FIRE前投資戦略スクリーニング（{len(pre_fire_investment_candidates)}候補）")
    best_pre_fire_strategy = {}
    best_pre_fire_df = None
    best_baseline_fire_month = None

    for i, pf_cand in enumerate(pre_fire_investment_candidates):
        cfg = copy.deepcopy(config)
        sim_cfg = cfg['simulation']
        for k, v in pf_cand.items():
            sim_cfg[k] = v
        cfg['simulation'] = sim_cfg

        pf_df = _simulate_pre_fire_trajectory(
            current_cash, current_stocks, cfg, scenario,
            monthly_income, monthly_expense
        )
        bl_df = simulate_future_assets(
            current_cash=current_cash,
            current_stocks=current_stocks,
            config=cfg,
            scenario=scenario,
            monthly_income=monthly_income,
            monthly_expense=monthly_expense,
        )
        fire_rows = bl_df[bl_df['fire_achieved'] == True]
        if len(fire_rows) == 0:
            if len(pre_fire_investment_candidates) > 1:
                pf_str = ', '.join(f'{k}={v}' for k, v in pf_cand.items()) if pf_cand else 'default'
                print(f"  [{i+1}/{len(pre_fire_investment_candidates)}] {pf_str}: FIRE不可")
            continue

        fm = int(fire_rows.iloc[0]['fire_month'])
        if len(pre_fire_investment_candidates) > 1:
            pf_str = ', '.join(f'{k}={v}' for k, v in pf_cand.items()) if pf_cand else 'default'
            print(f"  [{i+1}/{len(pre_fire_investment_candidates)}] {pf_str}: FIRE月={fm}（{start_age + fm/12:.1f}歳）")

        if best_baseline_fire_month is None or fm < best_baseline_fire_month:
            best_baseline_fire_month = fm
            best_pre_fire_df = pf_df
            best_pre_fire_strategy = pf_cand

    if best_baseline_fire_month is None:
        print("  [ERROR] 全候補でFIRE達成不可")
        return {'error': 'FIRE not achievable with any pre-FIRE investment strategy'}

    if len(pre_fire_investment_candidates) > 1 and best_pre_fire_strategy:
        pf_str = ', '.join(f'{k}={v}' for k, v in best_pre_fire_strategy.items())
        print(f"  最良FIRE前戦略: {pf_str} → FIRE月{best_baseline_fire_month}")

    pre_fire_df = best_pre_fire_df
    baseline_fire_month = best_baseline_fire_month

    # Phase 1
    if len(pre_fire_investment_candidates) <= 1:
        print("\nPhase 1: FIRE前シミュレーション（FIRE判定なし）...")
        pre_fire_df = _simulate_pre_fire_trajectory(
            current_cash, current_stocks, config, scenario,
            monthly_income, monthly_expense
        )
        print(f"  全{len(pre_fire_df)}ヶ月の状態を記録 [OK]")

        print("\n  基準FIRE月を取得中...")
        baseline_df = simulate_future_assets(
            current_cash=current_cash,
            current_stocks=current_stocks,
            config=config,
            scenario=scenario,
            monthly_income=monthly_income,
            monthly_expense=monthly_expense,
        )
        fire_rows = baseline_df[baseline_df['fire_achieved'] == True]
        if len(fire_rows) == 0:
            print("  [ERROR] 基準シミュレーションでFIRE達成不可")
            return {'error': 'FIRE not achievable in baseline simulation'}
        baseline_fire_month = int(fire_rows.iloc[0]['fire_month'])
    else:
        print(f"\nPhase 1: Phase 0の最良結果を使用（{len(pre_fire_df)}ヶ月分）")

    baseline_fire_age = start_age + baseline_fire_month / 12
    print(f"  基準FIRE月: {baseline_fire_month}（{baseline_fire_age:.1f}歳）")

    # Phase 2
    fire_month_min = max(1, baseline_fire_month - fire_month_search_range)
    fire_month_max = baseline_fire_month + fire_month_search_range
    max_month = len(pre_fire_df) - 1
    fire_month_max = min(fire_month_max, max_month)

    fire_month_candidates = list(range(fire_month_min, fire_month_max + 1, fire_month_step))
    if baseline_fire_month not in fire_month_candidates:
        fire_month_candidates.append(baseline_fire_month)
        fire_month_candidates.sort()

    # Phase 2a: 粗いグリッド（3年刻み）で高速スクリーニング
    coarse_ages = list(range(PENSION_AGE_MIN, PENSION_AGE_MAX + 1, 3))
    if PENSION_AGE_MAX not in coarse_ages:
        coarse_ages.append(PENSION_AGE_MAX)

    coarse_combos = len(coarse_ages) ** len(person_names)
    total_coarse = (len(fire_month_candidates) * coarse_combos
                    * len(extra_budget_candidates) * len(cash_strategy_candidates))

    print(f"\nPhase 2: 確定的スクリーニング")
    print(f"  FIRE候補月: {fire_month_candidates}")
    print(f"  年金開始年齢（粗いグリッド）: {coarse_ages} × {len(person_names)}人 = {coarse_combos}通り")
    if len(extra_budget_candidates) > 1:
        budget_str = ', '.join(f'{b/10000:.0f}万' for b in extra_budget_candidates)
        print(f"  追加予算候補: [{budget_str}]/月 ({len(extra_budget_candidates)}通り)")
    if len(cash_strategy_candidates) > 1:
        print(f"  現金管理戦略候補: {len(cash_strategy_candidates)}通り")
    print(f"  安全マージン: {min_baseline_final_assets/10000:.0f}万円")
    print(f"  合計候補数: {total_coarse}")

    if len(reduction_candidates) > 1:
        red_str = ', '.join(f'{r*100:.0f}%' for r in reduction_candidates)
        print(f"  緊縮削減率候補: [{red_str}]")
    total_coarse *= len(reduction_candidates)

    screening_results = _run_deterministic_screening(
        pre_fire_df, config, scenario,
        fire_month_candidates, coarse_ages, person_names,
        extra_budget_candidates=extra_budget_candidates,
        cash_strategy_candidates=cash_strategy_candidates,
        reduction_candidates=reduction_candidates,
        austerity_months=austerity_months,
    )

    feasible = screening_results[screening_results['final_assets'] > 0]
    print(f"  実行可能な候補: {len(feasible)}/{len(screening_results)}通り [OK]")

    if len(feasible) == 0:
        print("  [ERROR] 実行可能な候補なし")
        return {'error': 'No feasible candidates found'}

    feasible_sorted = feasible.sort_values(
        ['fire_month', 'extra_budget', 'final_assets'],
        ascending=[True, True, False]
    )

    coarse_top = _select_diverse_top_k(feasible_sorted, min(_REFINEMENT_TOP_K, len(feasible_sorted)), person_names)

    # Phase 2b: 有望な候補の近傍で細かいグリッド（1年刻み）にリファイン
    fine_fire_months = sorted(set(int(r['fire_month']) for _, r in coarse_top.iterrows()))
    fine_age_centers = set()
    for _, r in coarse_top.iterrows():
        for name in person_names:
            fine_age_centers.add(int(r[f'age_{name}']))

    fine_ages = set()
    for center in fine_age_centers:
        for delta in range(-2, 3):
            age = center + delta
            if PENSION_AGE_MIN <= age <= PENSION_AGE_MAX:
                fine_ages.add(age)
    fine_ages = sorted(fine_ages)

    fine_combos = len(fine_ages) ** len(person_names)
    total_fine = (len(fine_fire_months) * fine_combos * len(extra_budget_candidates)
                  * len(cash_strategy_candidates))
    print(f"\n  リファイン（1年刻み）: FIRE月{fine_fire_months}, 年金{fine_ages}")
    print(f"  リファイン候補数: {total_fine}")

    fine_results = _run_deterministic_screening(
        pre_fire_df, config, scenario,
        fine_fire_months, fine_ages, person_names,
        extra_budget_candidates=extra_budget_candidates,
        cash_strategy_candidates=cash_strategy_candidates,
        reduction_candidates=reduction_candidates,
        austerity_months=austerity_months,
    )

    all_screening = pd.concat([screening_results, fine_results], ignore_index=True)
    dedup_cols = (['fire_month', 'extra_budget', 'pre_pension_reduction']
                  + _CASH_STRATEGY_COLS
                  + [f'age_{n}' for n in person_names])
    all_screening = all_screening.drop_duplicates(subset=dedup_cols)

    feasible_mask = all_screening['final_assets'] >= min_baseline_final_assets
    if 'min_path_assets' in all_screening.columns and min_asset_floor > 0:
        feasible_mask = feasible_mask & (all_screening['min_path_assets'] >= min_asset_floor)
    feasible_count = len(all_screening[feasible_mask])
    floor_str = f"、最小パス資産>={min_asset_floor/10000:.0f}万円" if min_asset_floor > 0 else ""
    print(f"\n  安全マージン({min_baseline_final_assets/10000:.0f}万円){floor_str}以上: {feasible_count}/{len(all_screening)}通り")

    # 最適解を選出
    optimal, pareto_info = _find_optimal_solution(
        all_screening, min_baseline_final_assets, person_names,
        min_asset_floor=min_asset_floor,
    )

    # 基準解との比較情報を構築
    baseline_info = _get_baseline_info(
        baseline_fire_month, person_names, config
    )

    result = {
        'optimal': optimal,
        'baseline': baseline_info,
        'pareto_info': pareto_info,
        'min_baseline_final_assets': min_baseline_final_assets,
        'person_names': person_names,
        'pre_fire_strategy': best_pre_fire_strategy if best_pre_fire_strategy else None,
    }

    _print_result(result, config)

    return result


def _simulate_pre_fire_trajectory(
    current_cash: float,
    current_stocks: float,
    config: Dict[str, Any],
    scenario: str,
    monthly_income: float,
    monthly_expense: float
) -> pd.DataFrame:
    """Phase 1: FIRE判定なしで全期間の月次状態を記録する。"""
    df = simulate_future_assets(
        current_cash=current_cash,
        current_stocks=current_stocks,
        config=config,
        scenario=scenario,
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        disable_fire_check=True,
    )
    return df


def _compute_child_allowance_monthly_array(
    years_offset: float,
    remaining_months: int,
    config: Dict[str, Any],
) -> np.ndarray:
    """児童手当の月次配列を計算する（fire_monthごとに1回だけ呼び出す）。"""
    from src.simulator import calculate_child_allowance
    arr = np.empty(remaining_months)
    for m in range(remaining_months):
        arr[m] = calculate_child_allowance(years_offset + m / 12.0, config) / 12.0
    return arr


def _compute_pension_income_array_vectorized(
    years_offset: float,
    remaining_months: int,
    config: Dict[str, Any],
    post_fire_income: float,
    override_start_ages: Dict[str, int],
    child_allowance_monthly: np.ndarray,
) -> np.ndarray:
    """年金収入配列をベクトル化計算する（月ループを排除）。

    各人物の年金受給開始月を計算し、numpyスライスで配列を構築する。
    _calculate_person_pension を月数ではなく人数分（2回程度）だけ呼び出す。
    """
    import math
    from src.simulator import _calculate_person_pension, _get_age_at_offset

    pension_config = config['pension']
    default_start_age = pension_config['start_age']
    deferral_config = config['pension_deferral']
    increase_rate = deferral_config['deferral_increase_rate']
    decrease_rate = deferral_config['early_decrease_rate']
    pension_growth_rate = config['simulation']['standard']['pension_growth_rate']

    years_array = years_offset + np.arange(remaining_months) / 12.0
    if pension_growth_rate != 0.0:
        inflation_array = np.power(1.0 + pension_growth_rate, years_array)
    else:
        inflation_array = np.ones(remaining_months)

    # 年金収入ベース（インフレ前）: 各人の受給開始月以降に定額を加算
    pension_income_base = np.zeros(remaining_months)
    for person in pension_config['people']:
        person_name = person['name']
        birthdate_str = person.get('birthdate')
        if not birthdate_str:
            continue
        person_start_age = override_start_ages.get(person_name, default_start_age)

        # 各人物の実年齢から年金開始月を計算（start_age_config は修平の年齢であり桜とは異なる）
        person_age_at_fire = _get_age_at_offset(birthdate_str, years_offset)
        months_until = (person_start_age - person_age_at_fire) * 12.0
        start_month = max(0, math.ceil(months_until))
        if start_month >= remaining_months:
            continue

        # 年金ベース額を1回だけ計算（月ループなし）
        year_at_start = years_offset + start_month / 12.0
        base = _calculate_person_pension(
            person, year_at_start, person_start_age, True, years_offset
        )
        if base <= 0.0:
            continue

        age_diff = person_start_age - default_start_age
        if age_diff > 0:
            adj = 1.0 + increase_rate * age_diff
        elif age_diff < 0:
            adj = 1.0 - decrease_rate * abs(age_diff)
        else:
            adj = 1.0

        pension_income_base[start_month:] += base * adj / 12.0

    pension_income = pension_income_base * inflation_array
    labor_income = np.where(pension_income_base > 0, 0.0, post_fire_income)
    return pension_income + child_allowance_monthly + labor_income


def _screening_worker_fire_month(args: dict) -> List[dict]:
    """
    Phase 2確定的スクリーニング: 1つのfire_monthを処理するワーカー（並列化用）。

    ベクトル化した年金収入計算と児童手当の事前計算を使い、
    age_combos × extra_budgets × strategies の全組み合わせを評価して返す。
    """
    import numpy as np
    from src.simulator import (
        _simulate_post_fire_with_random_returns,
        _precompute_monthly_cashflows,
        calculate_child_allowance,
    )

    fire_month               = args['fire_month']
    pre_fire_row             = args['pre_fire_row']
    config                   = args['config']
    scenario                 = args['scenario']
    age_combos               = args['age_combos']
    person_names             = args['person_names']
    extra_budget_candidates  = args['extra_budget_candidates']
    cash_strategy_candidates = args['cash_strategy_candidates']
    monthly_return_rate      = args['monthly_return_rate']
    life_expectancy          = args['life_expectancy']
    start_age                = args['start_age']
    post_fire_income         = args['post_fire_income']
    default_target_reserve   = args['default_safety']
    default_crash            = args['default_crash']
    reduction_candidates     = args.get('reduction_candidates', [0.0])
    austerity_months         = args.get('austerity_months', 120)

    fire_cash        = pre_fire_row['cash']
    fire_stocks      = pre_fire_row['stocks']
    fire_nisa        = pre_fire_row['nisa_balance']
    fire_nisa_cost   = pre_fire_row.get('nisa_cost_basis', fire_nisa)
    fire_stocks_cost = pre_fire_row['stocks_cost_basis']
    years_offset     = fire_month / 12.0

    remaining_years  = life_expectancy - (start_age + years_offset)
    remaining_months = int(remaining_years * 12)
    if remaining_months <= 0:
        return []

    fixed_returns = np.full(remaining_months, monthly_return_rate)

    base_precomputed = _precompute_monthly_cashflows(
        years_offset, remaining_months, config, post_fire_income,
        override_start_ages=None
    )
    shared_expenses        = base_precomputed[0]
    shared_base_expenses   = base_precomputed[2]
    shared_life_stages     = base_precomputed[3]
    shared_workation_costs = base_precomputed[4]

    # 児童手当をfire_monthごとに1回だけ計算してキャッシュ
    child_allowance_monthly = _compute_child_allowance_monthly_array(
        years_offset, remaining_months, config
    )

    strategy_configs = [
        (cs, _apply_cash_strategy(config, cs))
        for cs in cash_strategy_candidates
    ]

    results = []
    for ages in age_combos:
        override = dict(zip(person_names, ages))

        # 緊縮期間終了月 = 最も早い年金受給開始年齢からの月数
        earliest_pension_age = min(ages)
        reduction_end_month_val = int((earliest_pension_age - start_age - years_offset) * 12)

        # ベクトル化版で月ループを排除
        income_array = _compute_pension_income_array_vectorized(
            years_offset, remaining_months, config,
            post_fire_income, override, child_allowance_monthly,
        )

        for extra_budget in extra_budget_candidates:
            for cs, cfg in strategy_configs:
                for reduction_rate in reduction_candidates:
                    final, min_path = _simulate_post_fire_with_random_returns(
                        current_cash=fire_cash,
                        current_stocks=fire_stocks,
                        years_offset=years_offset,
                        config=cfg,
                        scenario=scenario,
                        random_returns=fixed_returns,
                        nisa_balance=fire_nisa,
                        nisa_cost_basis=fire_nisa_cost,
                        stocks_cost_basis=fire_stocks_cost,
                        return_timeseries=False,
                        precomputed_expenses=shared_expenses,
                        precomputed_income=income_array,
                        precomputed_base_expenses=shared_base_expenses,
                        precomputed_life_stages=shared_life_stages,
                        precomputed_workation_costs=shared_workation_costs,
                        baseline_assets=None,
                        override_start_ages=override,
                        extra_monthly_budget=extra_budget,
                        return_min_path_assets=True,
                        pre_pension_reduction_rate=reduction_rate,
                        reduction_end_month=reduction_end_month_val,
                        austerity_months=austerity_months,
                    )

                    entry = {
                        'fire_month':              fire_month,
                        'extra_budget':            extra_budget,
                        'cash_target_reserve':     cs.get('target_cash_reserve', default_target_reserve),
                        'cash_crash_threshold':    cs.get('market_crash_threshold', default_crash),
                        'final_assets':            final,
                        'min_path_assets':         min_path,
                        'pre_pension_reduction':   reduction_rate,
                    }
                    for i, name in enumerate(person_names):
                        entry[f'age_{name}'] = ages[i]
                    results.append(entry)

    return results


def _run_deterministic_screening(
    pre_fire_df: pd.DataFrame,
    config: Dict[str, Any],
    scenario: str,
    fire_month_candidates: List[int],
    pension_ages: List[int],
    person_names: List[str],
    extra_budget_candidates: List[float] = None,
    cash_strategy_candidates: List[Dict[str, Any]] = None,
    reduction_candidates: List[float] = None,
    austerity_months: int = 120,
) -> pd.DataFrame:
    """
    Phase 2: 全候補を確定的シミュレーションで高速評価する。

    fire_monthレベルで ProcessPoolExecutor を使って並列化し、
    各ワーカー内では年金収入配列をベクトル化計算する。
    """
    if extra_budget_candidates is None:
        extra_budget_candidates = [0]
    if cash_strategy_candidates is None:
        cash_strategy_candidates = [{}]
    if reduction_candidates is None:
        reduction_candidates = [0.0]

    default_strategy = config['post_fire_cash_strategy']
    default_target_reserve = default_strategy['target_cash_reserve']
    default_crash = default_strategy['market_crash_threshold']

    params = config['simulation'][scenario]
    monthly_return_rate = (1 + params['annual_return_rate']) ** (1 / 12) - 1
    life_expectancy = config['simulation']['life_expectancy']
    start_age = config['simulation']['start_age']
    post_fire_income = (
        config['simulation']['shuhei_post_fire_income']
        + config['simulation']['sakura_post_fire_income']
    )

    age_combos = list(product(pension_ages, repeat=len(person_names)))

    # fire_monthごとの引数を準備
    all_args = []
    for fire_month in fire_month_candidates:
        if fire_month >= len(pre_fire_df):
            continue
        row = pre_fire_df.iloc[fire_month]
        all_args.append({
            'fire_month':               fire_month,
            'pre_fire_row':             row.to_dict(),
            'config':                   config,
            'scenario':                 scenario,
            'age_combos':               age_combos,
            'person_names':             person_names,
            'extra_budget_candidates':  extra_budget_candidates,
            'cash_strategy_candidates': cash_strategy_candidates,
            'reduction_candidates':     reduction_candidates,
            'austerity_months':         austerity_months,
            'monthly_return_rate':      monthly_return_rate,
            'life_expectancy':          life_expectancy,
            'start_age':                start_age,
            'post_fire_income':         post_fire_income,
            'default_safety':           default_target_reserve,
            'default_crash':            default_crash,
        })

    n_workers = min(multiprocessing.cpu_count(), len(all_args))
    n_total = len(all_args)
    all_results: List[dict] = []
    done = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(_PROJECT_ROOT,),
    ) as executor:
        futures = {
            executor.submit(_screening_worker_fire_month, args): args['fire_month']
            for args in all_args
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
            done += 1
            if done % max(1, n_total // 5) == 0 or done == n_total:
                print(f"  Progress: {done}/{n_total} fire_months完了")

    return pd.DataFrame(all_results)


def _select_diverse_top_k(
    feasible_sorted: pd.DataFrame,
    top_k: int,
    person_names: List[str]
) -> pd.DataFrame:
    """FIRE月×現金管理戦略の多様性を確保しつつ上位K件を選出する。

    (FIRE月, cash_target_reserve) の各グループからラウンドロビンで選出し、
    特定のFIRE月や特定の現金管理戦略に偏ることを防ぐ。
    """
    by_group = {}
    seen_combos = set()

    for _, row in feasible_sorted.iterrows():
        fm = int(row['fire_month'])
        ages = tuple(row[f'age_{name}'] for name in person_names)
        eb = row.get('extra_budget', 0)
        cs_reserve = row.get('cash_target_reserve', 0)
        cs_crash = row.get('cash_crash_threshold', 0)
        key = (fm, ages, eb, cs_reserve, cs_crash)
        if key in seen_combos:
            continue
        seen_combos.add(key)
        group_key = (fm, cs_reserve)
        by_group.setdefault(group_key, []).append(row)

    selected = []
    groups_sorted = sorted(by_group.keys())
    slot = 0
    while len(selected) < top_k:
        added = False
        for gk in groups_sorted:
            candidates = by_group[gk]
            if slot < len(candidates):
                selected.append(candidates[slot])
                added = True
                if len(selected) >= top_k:
                    break
        if not added:
            break
        slot += 1

    return pd.DataFrame(selected)



def _init_worker(project_root: str) -> None:
    """ワーカープロセスの sys.path を設定する（Windows spawn 対応）。"""
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _find_optimal_solution(
    screening_results: pd.DataFrame,
    min_baseline_final_assets: float,
    person_names: List[str],
    min_asset_floor: float = 0,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """ベースライン最終資産 >= 安全マージン かつ min_path_assets >= min_asset_floor の中で、FIRE月最小の解を選出する。"""
    has_min_path = 'min_path_assets' in screening_results.columns

    feasible = screening_results[screening_results['final_assets'] >= min_baseline_final_assets]
    if has_min_path and min_asset_floor > 0:
        feasible = feasible[feasible['min_path_assets'] >= min_asset_floor]

    # パレート: 各fire_monthで min_path_assets 最大の候補を選出
    pareto_candidates = []
    sort_col = 'min_path_assets' if has_min_path else 'final_assets'
    for fm in sorted(screening_results['fire_month'].unique()):
        fm_rows = screening_results[screening_results['fire_month'] == fm]
        best = fm_rows.sort_values(sort_col, ascending=False).iloc[0]
        pareto_candidates.append(best)

    pareto_df = pd.DataFrame(pareto_candidates)

    if len(feasible) == 0:
        return None, pareto_df

    feasible_sorted = feasible.sort_values(
        ['fire_month', sort_col],
        ascending=[True, False]
    )
    best = feasible_sorted.iloc[0]

    optimal = {
        'fire_month': int(best['fire_month']),
        'extra_monthly_budget': float(best.get('extra_budget', 0)),
        'cash_strategy': {
            'target_cash_reserve': float(best.get('cash_target_reserve', 3_000_000)),
            'market_crash_threshold': float(best.get('cash_crash_threshold', -0.20)),
        },
        'final_assets': float(best['final_assets']),
        'pension_ages': {name: int(best[f'age_{name}']) for name in person_names},
    }
    if has_min_path:
        optimal['min_path_assets'] = float(best['min_path_assets'])
    if 'pre_pension_reduction' in best.index:
        optimal['pre_pension_reduction'] = float(best['pre_pension_reduction'])
    return optimal, pareto_df


def _get_baseline_info(
    baseline_fire_month: int,
    person_names: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """現在のルールベースの結果を返す（比較用）。"""
    default_start_age = config['pension']['start_age']
    return {
        'fire_month': baseline_fire_month,
        'pension_ages': {name: default_start_age for name in person_names},
    }


def _print_result(result: Dict[str, Any], config: Dict[str, Any]) -> None:
    """最適化結果を整形して出力する。"""
    config_start_age = config['simulation']['start_age']
    pension_base_age = config['pension']['start_age']
    deferral_rate_pct = config['pension_deferral']['deferral_increase_rate'] * 100
    early_rate_pct = config['pension_deferral']['early_decrease_rate'] * 100

    print("\n" + "=" * 60)
    print("最適化結果")
    print("=" * 60)

    optimal = result.get('optimal')
    baseline = result.get('baseline')
    person_names = result.get('person_names', [])
    min_bl = result.get('min_baseline_final_assets', DEFAULT_MIN_BASELINE_FINAL_ASSETS)

    if optimal is None:
        print(f"\n  ベースライン最終資産 >= {min_bl/10000:.0f}万円 を満たす解が見つかりませんでした。")
        print("  安全マージンを下げるか、探索範囲を広げてください。")
    else:
        fire_age = config_start_age + optimal['fire_month'] / 12
        extra_budget = optimal.get('extra_monthly_budget', 0)
        cs = optimal.get('cash_strategy', {})
        print(f"\n  最適解:")
        print(f"    FIRE達成月:         月{optimal['fire_month']}（{fire_age:.1f}歳）")
        for name in person_names:
            age = optimal['pension_ages'][name]
            diff = age - pension_base_age
            if diff > 0:
                adj_str = f"+{diff*deferral_rate_pct:.1f}%増額"
            elif diff < 0:
                adj_str = f"{diff*early_rate_pct:.1f}%減額"
            else:
                adj_str = "増減なし"
            print(f"    {name}の受給開始年齢:  {age}歳（{adj_str}）")
        if extra_budget > 0:
            print(f"    FIRE後追加予算:     月{extra_budget/10000:.0f}万円（年{extra_budget*12/10000:.0f}万円）")
        if cs:
            print(f"    現金確保目標:        {cs.get('target_cash_reserve', 3_000_000)/10000:.0f}万円")
            print(f"    暴落判定閾値:       {cs.get('market_crash_threshold', -0.20)*100:.0f}%")
        final_assets = optimal.get('final_assets', 0)
        min_path = optimal.get('min_path_assets')
        reduction = optimal.get('pre_pension_reduction', 0)
        print(f"    ベースライン最終資産: {final_assets/10000:.0f}万円（安全マージン: {min_bl/10000:.0f}万円）")
        if min_path is not None:
            print(f"    最小パス資産:       {min_path/10000:.0f}万円")
        if reduction > 0:
            print(f"    年金前緊縮削減率:   {reduction*100:.0f}%")

        if baseline:
            bl_fire_age = config_start_age + baseline['fire_month'] / 12
            month_diff = baseline['fire_month'] - optimal['fire_month']
            print(f"\n  現在のルールベースとの比較:")
            print(f"    ルールベースFIRE月:  月{baseline['fire_month']}（{bl_fire_age:.1f}歳）")
            for name in person_names:
                print(f"    ルールベース年金({name}): {baseline['pension_ages'][name]}歳")
            if month_diff > 0:
                print(f"    [最適化] FIRE時期を {month_diff}ヶ月前倒し")
            elif month_diff < 0:
                print(f"    [最適化] FIRE時期は {-month_diff}ヶ月後ろ倒し（安全マージン制約のため）")
            else:
                print(f"    [最適化] FIRE時期は同一（年金戦略の最適化のみ）")

    # パレート参考情報
    pareto = result.get('pareto_info')
    if pareto is not None and len(pareto) > 0:
        print(f"\n" + "=" * 60)
        print("パレート参考情報（FIRE月 vs ベースライン最終資産）")
        print("=" * 60)

        ages_header = '/'.join(person_names)
        print(f"\n  {'FIRE月':>6} | {'年齢':>5} | {f'年金({ages_header})':>16} | {'追加予算':>8} | {'最終資産':>10}")
        print(f"  {'-'*6} | {'-'*5} | {'-'*16} | {'-'*8} | {'-'*10}")

        for _, row in pareto.iterrows():
            fm = int(row['fire_month'])
            age = config_start_age + fm / 12
            ages_str = '/'.join(f"{int(row[f'age_{n}'])}歳" for n in person_names)
            eb = float(row.get('extra_budget', 0))
            eb_str = f"{eb/10000:.0f}万/月" if eb > 0 else "-"
            fa = float(row.get('final_assets', 0))
            is_optimal = (optimal
                          and fm == optimal['fire_month']
                          and eb == optimal.get('extra_monthly_budget', 0))
            marker = " ★" if is_optimal else ""
            meets = "  " if fa >= min_bl else "x "
            print(f"  {meets}月{fm:>3} | {age:>4.1f}歳 | {ages_str:>16} | {eb_str:>8} | {fa/10000:>8.0f}万円{marker}")
