"""
シミュレーターモジュール
将来の資産推移を予測
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass, field
from dateutil.relativedelta import relativedelta
from functools import lru_cache

# 年金・資産計算用定数
_NATIONAL_PENSION_FULL_AMOUNT = 816_000       # 国民年金満額（2024年度, 円/年）
_EMPLOYEE_PENSION_MULTIPLIER = 0.005481        # 厚生年金乗率（給付乗率）
_PENSION_MAX_CONTRIBUTION_YEARS = 40           # 国民年金最大加入年数

# 国民健康保険料計算用定数（config未設定時のフォールバック）
_HEALTH_INS_BASIC_DEDUCTION = 430_000  # 基礎控除（2024年度・円）
_HEALTH_INS_DEFAULT_INCOME_RATE = 0.11  # 所得割率デフォルト（40歳以上・全国平均）

# シミュレーション参照日付（再現性のため実行ごとに一度だけ設定）
_REFERENCE_DATE: datetime = None


def _set_reference_date(date: datetime = None) -> None:
    """シミュレーション開始時に参照日付を固定する。"""
    global _REFERENCE_DATE
    _REFERENCE_DATE = date or datetime.now()


def _get_reference_date() -> datetime:
    """固定された参照日付を返す。未設定時は現在時刻で初期化。"""
    global _REFERENCE_DATE
    if _REFERENCE_DATE is None:
        _REFERENCE_DATE = datetime.now()
    return _REFERENCE_DATE


class _StockSaleResult(NamedTuple):
    """_sell_stocks_with_tax() の戻り値。"""
    stocks: float            # 更新後の株式残高
    nisa_balance: float      # 更新後のNISA残高
    nisa_cost_basis: float   # 更新後のNISA簿価
    stocks_cost_basis: float # 更新後の株式簿価
    nisa_sold: float         # NISA売却額（非課税）
    cash_from_taxable: float # 課税口座売却後の現金（税引後）
    capital_gain: float      # 実現益（capital_gains_this_year に加算する値）
    total_sold: float        # 総売却額（nisa_sold + 課税口座売却額）


@dataclass
class SimulationState:
    """
    シミュレーションの状態を保持するクラス

    月次シミュレーション処理の共通化のため、各シミュレーション関数が
    共有する状態変数を一つのクラスにまとめる。
    """
    # 時間
    month_index: int
    current_date: datetime
    year_offset: float

    # 資産
    cash: float
    stocks: float
    nisa_balance: float
    stocks_cost_basis: float
    nisa_cost_basis: float

    # 年度管理
    current_year: int
    nisa_annual_invested: float = 0.0

    # 譲渡益
    capital_gains_this_year: float = 0.0
    prev_year_capital_gains: float = 0.0

    # FIRE関連
    fire_achieved: bool = False
    fire_achievement_month: Optional[int] = None

    # 設定
    config: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)  # シナリオパラメータ


class SimulationCallbacks:
    """
    シミュレーション固有の処理を定義するコールバッククラス

    各シミュレーション関数は、このクラスを継承して固有の処理を実装する。
    """

    def on_month_start(self, state: SimulationState) -> None:
        """
        月初処理（オプション）

        Args:
            state: シミュレーション状態（破壊的に更新可能）
        """
        pass

    def on_month_end(self, state: SimulationState) -> Optional[Dict[str, Any]]:
        """
        月末処理（オプション）

        Args:
            state: シミュレーション状態（破壊的に更新可能）

        Returns:
            結果レコードに追加するデータ（オプション）
        """
        return None

    def should_terminate(self, state: SimulationState) -> bool:
        """
        シミュレーション終了判定

        Args:
            state: シミュレーション状態

        Returns:
            True ならシミュレーション終了
        """
        return False


def _get_monthly_return_rate(annual_return_rate: float) -> float:
    """
    年率リターンから月次リターン率を計算（複利）。

    Args:
        annual_return_rate: 年率リターン（例: 0.05 = 5%）

    Returns:
        月次リターン率
    """
    return (1 + annual_return_rate) ** (1/12) - 1


def generate_random_returns(
    annual_return_mean: float,
    annual_return_std: float,
    total_months: int,
    mean_reversion_speed: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    月次リターンをランダム生成（対数正規分布 + 平均回帰）

    モンテカルロシミュレーション用に、市場変動を考慮した
    ランダムな月次リターン率を生成する。

    対数正規分布モデルを採用することで：
    - 資産価格が負にならない（理論的に正確）
    - 幾何平均が自然に保存される
    - 手動のVolatility Drag補正が不要

    平均回帰モデル（AR型）を追加することで：
    - 大幅な下落後は反発しやすくなる（現実的）
    - 大幅な上昇後は調整が入りやすくなる
    - 連続的な暴落シナリオを抑制
    - 実際の市場データと整合的な挙動

    Args:
        annual_return_mean: 年率リターン平均（例: 0.05 = 5%）
        annual_return_std: 年率リターン標準偏差（例: 0.06 = 6%）
        total_months: シミュレーション月数
        mean_reversion_speed: 平均回帰の速度（0.0-1.0）
            0.0 = 回帰なし（独立な乱数）
            0.3 = 緩やかな回帰（推奨、現実的な市場）
            0.5 = 中程度の回帰
            1.0 = 完全回帰（非現実的）
        random_seed: 乱数シード（再現性のため、オプション）

    Returns:
        月次リターン率の配列（長さ: total_months）

    Example:
        >>> # 平均回帰なし（従来の挙動）
        >>> returns = generate_random_returns(0.05, 0.06, 12, mean_reversion_speed=0.0)
        >>> len(returns)
        12
        
        >>> # 平均回帰あり（推奨）
        >>> returns = generate_random_returns(0.05, 0.06, 12, mean_reversion_speed=0.3, random_seed=42)
        >>> returns[0] > -1  # リターンは-100%未満にならない
        True
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 対数リターンのパラメータ計算
    # 年率リターンを対数空間に変換
    log_annual_mean = np.log(1 + annual_return_mean)
    log_monthly_mean = log_annual_mean / 12

    # 標準偏差は√12で月次に変換
    log_monthly_std = annual_return_std / np.sqrt(12)

    # 平均回帰モデル（AR型）
    if mean_reversion_speed > 0:
        returns = np.zeros(total_months)
        prev_log_return = log_monthly_mean  # 初期値は平均

        for t in range(total_months):
            # 平均回帰項: 前月が平均を下回った→次月は上振れしやすい
            deviation = prev_log_return - log_monthly_mean
            mean_reversion_adjustment = -mean_reversion_speed * deviation

            # 今月のリターンを生成（平均回帰バイアス付き）
            log_return = np.random.normal(
                loc=log_monthly_mean + mean_reversion_adjustment,
                scale=log_monthly_std
            )

            # 算術リターンに変換
            returns[t] = np.exp(log_return) - 1
            prev_log_return = log_return

        return returns
    else:
        # 平均回帰なし（従来の独立な乱数生成）
        log_returns = np.random.normal(
            loc=log_monthly_mean,
            scale=log_monthly_std,
            size=total_months
        )

        # 算術リターンに変換
        # r = exp(log_r) - 1
        returns = np.exp(log_returns) - 1

        return returns


def generate_random_returns_batch(
    annual_return_mean: float,
    annual_return_std: float,
    total_months: int,
    n_paths: int,
    mean_reversion_speed: float = 0.0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    月次リターンのバッチ生成（generate_random_returns の N パス同時処理版）

    mean_reversion_speed=0 の場合は完全ベクトル化（ループなし）。
    mean_reversion_speed>0 の場合は T ループ + N ベクトル演算。

    Args:
        annual_return_mean: 年率リターン平均
        annual_return_std: 年率リターン標準偏差
        total_months: シミュレーション月数
        n_paths: 同時生成するパス数
        mean_reversion_speed: 平均回帰の速度（0.0-1.0）
        random_seed: 乱数シード

    Returns:
        shape (n_paths, total_months) の月次リターン率行列
    """
    rng = np.random.default_rng(random_seed)
    log_monthly_mean = np.log(1 + annual_return_mean) / 12
    log_monthly_std = annual_return_std / np.sqrt(12)

    if mean_reversion_speed > 0:
        eps = rng.standard_normal((n_paths, total_months))
        returns = np.zeros((n_paths, total_months))
        prev_log = np.full(n_paths, log_monthly_mean)  # [N]

        for t in range(total_months):
            deviation = prev_log - log_monthly_mean  # [N]
            adjustment = -mean_reversion_speed * deviation  # [N]
            log_ret = log_monthly_mean + adjustment + log_monthly_std * eps[:, t]  # [N]
            returns[:, t] = np.exp(log_ret) - 1
            prev_log = log_ret

        return returns
    else:
        log_returns = rng.normal(
            loc=log_monthly_mean,
            scale=log_monthly_std,
            size=(n_paths, total_months),
        )
        return np.exp(log_returns) - 1

def generate_returns_enhanced(
    annual_return_mean: float,
    annual_return_std: float,
    total_months: int,
    config: Dict[str, Any],
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    拡張リターン生成（GARCH + 非対称多期間平均回帰）

    現実的な市場ダイナミクスをモデル化:
    - ボラティリティ・クラスタリング（暴落後の高ボラティリティ持続）
    - 暴落後の遅い回復（重度の弱気相場は5年以上かけて回復）
    - バブル後の速い収縮（非対称）
    - レジーム持続性（弱気相場は18-24ヶ月持続）

    数学モデル:
        ステップ1: GARCH(1,1)ボラティリティ
            σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²

        ステップ2: 多期間平均回帰
            R_cum = 過去12ヶ月の累積リターン
            deviation = R_cum - 期待累積リターン
            
            λ = {
                λ_crash  if deviation < -15% (暴落からの回復)
                λ_bubble if deviation > +15% (バブルの収縮)
                λ_normal otherwise          (通常の回帰)
            }
            
            adjustment = -λ × deviation / window

        ステップ3: リターン生成
            log(r_t) = μ + adjustment + σ_t × ε_t
            r_t = exp(log(r_t)) - 1

    Args:
        annual_return_mean: 年率期待リターン（例: 0.05 = 5%）
        annual_return_std: 年率ボラティリティ（例: 0.06 = 6%）
        total_months: シミュレーション期間（月）
        config: 設定辞書（monte_carlo.enhanced_model）
        random_seed: 乱数シード（再現性のため）

    Returns:
        月次リターン率の配列（長さ: total_months）

    Example:
        >>> config = {'simulation': {'monte_carlo': {'enhanced_model': {...}}}}
        >>> returns = generate_returns_enhanced(0.05, 0.06, 120, config, seed=42)
        >>> len(returns)
        120
        >>> # ボラティリティ・クラスタリングを確認
        >>> abs_returns = np.abs(returns)
        >>> autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        >>> autocorr > 0.05  # ボラティリティは正の自己相関を持つ
        True
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 設定の取得
    mc_config = config['simulation']['monte_carlo']
    enhanced = mc_config['enhanced_model']

    # GARCHパラメータ
    ω = enhanced['garch_omega']
    α = enhanced['garch_alpha']
    β = enhanced['garch_beta']
    σ_floor = enhanced['volatility_floor']
    σ_ceil = enhanced['volatility_ceiling']

    # 平均回帰パラメータ
    mr_window = enhanced['mean_reversion_window']
    λ_crash = enhanced['mr_speed_crash']
    λ_normal = enhanced['mr_speed_normal']
    λ_bubble = enhanced['mr_speed_bubble']
    crash_thresh = enhanced['crash_threshold']
    bubble_thresh = enhanced['bubble_threshold']

    # 年率から月次へ変換（対数空間）
    μ_monthly = np.log(1 + annual_return_mean) / 12
    σ_long = annual_return_std / np.sqrt(12)

    # 初期化
    returns = np.zeros(total_months)
    log_returns = np.zeros(total_months)
    σ_t = σ_long  # 初期ボラティリティ = 長期平均

    for t in range(total_months):
        # 平均回帰調整（多期間ルックバック）
        if t >= mr_window:
            # 過去mr_window月の累積リターンを計算
            R_cum = np.prod(1 + returns[t-mr_window:t]) - 1
            # 期待累積リターン
            R_expected = (1 + annual_return_mean) ** (mr_window/12) - 1
            # 偏差
            R_deviation = R_cum - R_expected

            # 非対称平均回帰速度の選択
            if R_deviation < crash_thresh:
                # 暴落からの回復: 遅い（現実的な5年回復）
                λ = λ_crash
            elif R_deviation > bubble_thresh:
                # バブル崩壊: 最も遅い（バブルは持続する）
                λ = λ_bubble
            else:
                # 通常: 現在の実装
                λ = λ_normal

            # 月次への調整量
            mr_adjustment = -λ * R_deviation / mr_window
        else:
            mr_adjustment = 0.0

        # ショックの生成
        ε_t = np.random.standard_normal()

        # 平均回帰を適用した対数リターン
        log_ret = μ_monthly + mr_adjustment + σ_t * ε_t
        log_returns[t] = log_ret
        returns[t] = np.exp(log_ret) - 1

        # GARCHボラティリティ更新
        # σ_t² = ω + α·(σ_t·ε_t)² + β·σ_t²
        realized_shock_squared = (σ_t * ε_t) ** 2
        σ_t_squared = ω + α * realized_shock_squared + β * σ_t ** 2

        # ボラティリティのクリッピング（暴走防止 / 非現実的な静穏防止）
        σ_t = np.sqrt(np.clip(σ_t_squared, σ_floor**2, σ_ceil**2))

    return returns


def generate_returns_enhanced_batch(
    annual_return_mean: float,
    annual_return_std: float,
    total_months: int,
    n_paths: int,
    config: Dict[str, Any],
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    拡張リターンのバッチ生成（N パス同時処理、高速化版）

    generate_returns_enhanced を N パス同時に生成することで、
    Python for ループの呼び出し回数を 1/N に削減する。

    時系列依存（GARCH の σ_t、平均回帰の累積リターン）は
    T 回のループで各ステップをベクトル演算として処理する。
    ループ回数: N×T → T（N パスを同時処理するため）

    Args:
        annual_return_mean: 年率期待リターン
        annual_return_std: 年率ボラティリティ
        total_months: シミュレーション期間（月）
        n_paths: 同時生成するパス数
        config: 設定辞書（monte_carlo.enhanced_model）
        random_seed: 乱数シード（再現性のため）

    Returns:
        shape (n_paths, total_months) の月次リターン率行列
    """
    rng = np.random.default_rng(random_seed)
    eps = rng.standard_normal((n_paths, total_months))  # 全ε_t を事前生成

    mc_config = config['simulation']['monte_carlo']
    enhanced = mc_config['enhanced_model']

    ω = enhanced['garch_omega']
    α = enhanced['garch_alpha']
    β = enhanced['garch_beta']
    σ_floor = enhanced['volatility_floor']
    σ_ceil = enhanced['volatility_ceiling']

    mr_window = enhanced['mean_reversion_window']
    λ_crash = enhanced['mr_speed_crash']
    λ_normal = enhanced['mr_speed_normal']
    λ_bubble = enhanced['mr_speed_bubble']
    crash_thresh = enhanced['crash_threshold']
    bubble_thresh = enhanced['bubble_threshold']

    μ_monthly = np.log(1 + annual_return_mean) / 12
    σ_long = annual_return_std / np.sqrt(12)
    R_expected = (1 + annual_return_mean) ** (mr_window / 12) - 1

    returns = np.zeros((n_paths, total_months))
    σ_t = np.full(n_paths, σ_long)  # [N]

    for t in range(total_months):
        if t >= mr_window:
            R_cum = np.prod(1 + returns[:, t - mr_window:t], axis=1) - 1  # [N]
            R_deviation = R_cum - R_expected  # [N]
            λ = np.where(
                R_deviation < crash_thresh, λ_crash,
                np.where(R_deviation > bubble_thresh, λ_bubble, λ_normal),
            )  # [N]
            mr_adjustment = -λ * R_deviation / mr_window  # [N]
        else:
            mr_adjustment = 0.0

        log_ret = μ_monthly + mr_adjustment + σ_t * eps[:, t]  # [N]
        returns[:, t] = np.exp(log_ret) - 1

        realized_shock_sq = (σ_t * eps[:, t]) ** 2  # [N]
        σ_t_sq = ω + α * realized_shock_sq + β * σ_t ** 2  # [N]
        σ_t = np.sqrt(np.clip(σ_t_sq, σ_floor ** 2, σ_ceil ** 2))  # [N]

    return returns


def calculate_drawdown_level(
    current_assets: float,
    peak_assets_history: List[float],
    config: Dict[str, Any],
    planned_assets: float = None,
) -> Tuple[float, int]:
    """
    ベースラインとの乖離率を計算する。

    比例制御型動的支出調整のために、計画資産額に対する現在の乖離率を返す。
    level は後方互換性のため残すが、常に 0 を返す（比例制御では不使用）。

    Args:
        current_assets: 現在の総資産（円）
        peak_assets_history: 過去の資産履歴（planned_assets未指定時に使用）
        config: 設定辞書
        planned_assets: 計画資産額（円）。指定時はこちらを基準とする。

    Returns:
        (drawdown, 0):
            - drawdown: 乖離率（正=上ぶれ、負=下ぶれ）
            - 0: 後方互換性のため常に0
    """
    if planned_assets is not None and planned_assets > 0:
        reference = planned_assets
    else:
        reference = max(peak_assets_history) if peak_assets_history else current_assets

    drawdown = (current_assets / reference - 1.0) if reference > 0 else 0.0
    return drawdown, 0


def calculate_proportional_expense_adjustment(
    surplus: float,
    discretionary_monthly: float,
    config: Dict[str, Any],
) -> float:
    """
    比例制御型の動的支出調整額を計算する。

    ベースラインとの乖離額（surplus）に比例して支出を調整する。
    余剰（surplus > 0）→ 支出増加、不足（surplus < 0）→ 支出削減。
    調整額は裁量支出の倍率でキャップされる。

    Args:
        surplus: ベースラインとの乖離額（actual - baseline、円）
        discretionary_monthly: 月次裁量支出（円）
        config: 設定辞書

    Returns:
        月次支出調整額（正=支出増加、負=支出削減）
    """
    dynamic_config = config['fire']['dynamic_expense_reduction']
    if not dynamic_config['enabled']:
        return 0.0

    alpha = dynamic_config['surplus_spending_rate']
    max_cut_ratio = dynamic_config['max_cut_ratio']
    max_boost_ratio = dynamic_config['max_boost_ratio']

    monthly_adjustment = surplus * alpha / 12.0

    max_boost = discretionary_monthly * max_boost_ratio
    max_cut = discretionary_monthly * max_cut_ratio

    return max(-max_cut, min(monthly_adjustment, max_boost))


@lru_cache(maxsize=32)
def _parse_birthdate(birthdate_str: str) -> datetime:
    """生年月日文字列をパース（キャッシュ付き）"""
    return datetime.strptime(birthdate_str, '%Y/%m/%d')

def _get_age_at_offset(birthdate_str: str, year_offset: float) -> float:
    """
    生年月日文字列とシミュレーション経過年数から、その時点での年齢を返す。

    Args:
        birthdate_str: 生年月日（'YYYY/MM/DD'形式）
        year_offset: シミュレーション開始からの経過年数

    Returns:
        シミュレーション時点での年齢（歳）
    """
    birthdate = _parse_birthdate(birthdate_str)  # キャッシュされたパース結果を使用
    current_age = (_get_reference_date() - birthdate).days / 365.25
    return current_age + year_offset


def _get_primary_parent_age(config: Dict[str, Any], year_offset: float) -> Optional[float]:
    """
    設定から主たる親（年金設定の最初の人）の指定年時点での年齢を返す。
    シミュレーション中の empty_nest サブステージ判定に使用する。
    """
    pension_people = config['pension']['people']
    if pension_people:
        birthdate_str = pension_people[0]['birthdate']
        if birthdate_str:
            return _get_age_at_offset(birthdate_str, year_offset)
    start_age = config['simulation']['start_age']
    if start_age is not None:
        return float(start_age) + year_offset
    return None


def _get_life_stage(
    child_age: float,
    parent_age: Optional[float] = None,
    senior_from_age: int = 70,
    elderly_from_age: int = 80,
) -> str:
    """
    子供の年齢からライフステージキーを返す。
    child_age >= 22（子供独立後）の場合、parent_age に応じてサブステージを返す。

    Returns:
        'young_child' | 'elementary' | 'junior_high' | 'high_school' | 'university'
        | 'empty_nest_active' | 'empty_nest_senior' | 'empty_nest_elderly'
        | 'empty_nest' (parent_age が不明な場合の後方互換)
    """
    if child_age < 6:
        return 'young_child'
    elif child_age < 12:
        return 'elementary'
    elif child_age < 15:
        return 'junior_high'
    elif child_age < 18:
        return 'high_school'
    elif child_age < 22:
        return 'university'
    else:
        if parent_age is None:
            return 'empty_nest'
        if parent_age >= elderly_from_age:
            return 'empty_nest_elderly'
        elif parent_age >= senior_from_age:
            return 'empty_nest_senior'
        else:
            return 'empty_nest_active'


def _is_enabled(config: Dict, section: str) -> bool:
    """設定セクションの enabled フラグを返す。"""
    return config[section]['enabled']


def _advance_year(
    date_year: int,
    current_year: int,
    capital_gains_this_year: float,
) -> tuple:
    """
    月次ループで年が変わった場合に年次カウンターをリセットする。

    Returns:
        (new_current_year, new_capital_gains_this_year, year_advanced: bool)
        year_advanced=True の場合、呼び出し側で capital_gains_this_year の元の値を
        prev_year_capital_gains として保存すること。
    """
    if date_year > current_year:
        return date_year, 0.0, True
    return current_year, capital_gains_this_year, False


def calculate_education_expense(year_offset: float, config: Dict[str, Any]) -> float:
    """
    指定年における教育費を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書

    Returns:
        年間教育費（円）
    """
    if not _is_enabled(config, 'education'):
        return 0

    children = config['education']['children']
    costs = config['education']['costs']

    total_education_expense = 0

    for child in children:
        # 生年月日から現在の年齢を計算
        birthdate_str = child['birthdate']
        if not birthdate_str:
            continue

        child_age = _get_age_at_offset(birthdate_str, year_offset)

        # 年齢に応じた教育段階と費用を計算
        if 0 <= child_age < 3:
            # 保育園（0-2歳）
            nursery_type = child['nursery']
            if nursery_type != 'none':
                annual_cost = costs['nursery'].get(nursery_type, 0)
                total_education_expense += annual_cost

        elif 3 <= child_age < 6:
            # 幼稚園（3-5歳）
            stage_type = child['kindergarten']
            annual_cost = costs['kindergarten'][stage_type]
            total_education_expense += annual_cost

        elif 6 <= child_age < 12:
            # 小学校（6-11歳）
            stage_type = child['elementary']
            annual_cost = costs['elementary'][stage_type]
            total_education_expense += annual_cost

        elif 12 <= child_age < 15:
            # 中学校（12-14歳）
            stage_type = child['junior_high']
            annual_cost = costs['junior_high'][stage_type]
            total_education_expense += annual_cost

        elif 15 <= child_age < 18:
            # 高校（15-17歳）
            stage_type = child['high']
            annual_cost = costs['high'][stage_type]
            total_education_expense += annual_cost

        elif 18 <= child_age < 22:
            # 大学（18-21歳）
            stage_type = child['university']
            annual_cost = costs['university'][stage_type]
            total_education_expense += annual_cost

    return total_education_expense


def _calculate_national_pension_amount(contribution_years: float) -> float:
    """
    国民年金の年間受給額を計算

    Args:
        contribution_years: 加入年数（最大40年）

    Returns:
        年間年金額（円）
    """
    # 加入年数に応じた按分（最大加入年数で満額）
    contribution_years = min(contribution_years, _PENSION_MAX_CONTRIBUTION_YEARS)
    annual_pension = _NATIONAL_PENSION_FULL_AMOUNT * (contribution_years / _PENSION_MAX_CONTRIBUTION_YEARS)

    return annual_pension


def _calculate_employees_pension_amount(avg_monthly_salary: float, contribution_months: int) -> float:
    """
    厚生年金の年間受給額を計算（簡易版）

    Args:
        avg_monthly_salary: 平均月収（円）- 厚生年金加入期間中の平均
        contribution_months: 加入月数

    Returns:
        年間年金額（円）
    """
    # 厚生年金（報酬比例部分）= 平均月収 × 加入月数 × 給付乗率
    # 2003年4月以降の乗率（平成15年3月以前と以降で異なるが、簡略化のため統一）
    annual_pension = avg_monthly_salary * contribution_months * _EMPLOYEE_PENSION_MULTIPLIER

    return annual_pension


def _calculate_person_pension(
    person: dict,
    year_offset: float,
    start_age: float,
    fire_achieved: bool,
    fire_year_offset: float,
) -> float:
    """
    1人分の年金年間受給額を計算する。

    Args:
        person: config['pension']['people'] の1エントリ
        year_offset: シミュレーション開始からの経過年数
        start_age: 年金受給開始年齢
        fire_achieved: FIRE達成フラグ
        fire_year_offset: FIRE達成時点の経過年数（年）

    Returns:
        年間年金受給額（円）。受給開始前は 0。
    """
    birthdate_str = person['birthdate']
    if not birthdate_str:
        return 0

    person_age = _get_age_at_offset(birthdate_str, year_offset)
    if person_age < start_age:
        return 0

    pension_type = person['pension_type']

    if pension_type == 'employee':
        avg_monthly_salary = person.get('avg_monthly_salary', 0)
        work_start_age = person.get('work_start_age', 23)
        past_pension_base = person.get('past_pension_base_annual', 0)
        past_months = person.get('past_contribution_months', 0)
        if fire_achieved and fire_year_offset is not None:
            work_end_age = _get_age_at_offset(birthdate_str, fire_year_offset)
        else:
            work_end_age = min(person_age, 65)
        work_months = int(max(0, work_end_age - work_start_age) * 12)
        if past_pension_base > 0 and past_months > 0:
            # 実績ベース: ねんきん定期便の過去実績額 + FIREまでの将来積み上げ
            future_months = max(0, work_months - past_months)
            employees_pension = past_pension_base + avg_monthly_salary * future_months * _EMPLOYEE_PENSION_MULTIPLIER
        else:
            employees_pension = _calculate_employees_pension_amount(avg_monthly_salary, work_months)
        national_pension = _calculate_national_pension_amount(_PENSION_MAX_CONTRIBUTION_YEARS)
        return employees_pension + national_pension

    elif pension_type == 'national':
        return _calculate_national_pension_amount(_PENSION_MAX_CONTRIBUTION_YEARS)

    else:
        # 従来の固定値フォールバック（後方互換性）
        return person.get('annual_amount', 0)


def _extract_override_start_ages(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """
    config.yaml の pension.people[].override_start_age から
    override_start_ages 辞書を構築する。
    1人でも override_start_age が指定されていれば辞書を返す。
    誰も指定していなければ None を返す。
    """
    pension_config = config['pension']
    people = pension_config['people']
    ages = {}
    for person in people:
        name = person['name']
        override_age = person.get('override_start_age')
        if override_age is not None:
            ages[name] = int(override_age)
    return ages if ages else None


def _determine_optimal_pension_start_age(
    current_assets: float,
    config: Dict[str, Any],
    fire_target_assets: float = None
) -> int:
    """
    資産状況に基づいて最適な年金受給開始年齢を決定する。

    Args:
        current_assets: 現在の総資産（円）
        config: 設定辞書
        fire_target_assets: FIRE目標資産（円、Noneの場合は計算）

    Returns:
        最適な年金受給開始年齢
    """
    deferral_config = config['pension_deferral']

    if not deferral_config['enabled']:
        # 年金繰り下げ戦略が無効の場合、従来の固定年齢を使用
        return config['pension']['start_age']

    # FIRE目標資産を計算（未提供の場合）
    if fire_target_assets is None:
        # 簡易計算: 年間基礎支出 / 4% ルール
        base_expense = config['fire']['base_expense_by_stage']['empty_nest']
        fire_target_assets = base_expense / 0.04

    # 資産比率を計算（現在資産 / FIRE目標資産）
    asset_ratio = current_assets / fire_target_assets if fire_target_assets > 0 else 0

    # 閾値を取得
    defer_to_70_threshold = deferral_config['defer_to_70_threshold']
    defer_to_68_threshold = deferral_config['defer_to_68_threshold']
    early_at_62_threshold = deferral_config['early_at_62_threshold']

    # 最適年齢を決定
    if asset_ratio >= defer_to_70_threshold:
        # 資産が目標の150%以上 → 70歳まで繰り下げ
        return 70
    elif asset_ratio >= defer_to_68_threshold:
        # 資産が目標の120%以上 → 68歳まで繰り下げ
        return 68
    elif asset_ratio < early_at_62_threshold:
        # 資産が目標の50%未満 → 62歳で繰り上げ受給
        return 62
    else:
        # 通常通り65歳から受給
        return 65


def calculate_pension_income(
    year_offset: float,
    config: Dict[str, Any],
    fire_achieved: bool = False,
    fire_year_offset: float = None,
    current_assets: float = None,
    fire_target_assets: float = None,
    override_start_ages: Dict[str, int] = None
) -> float:
    """
    指定年における年金収入を計算（FIRE対応・動的計算・繰り下げ戦略対応）

    FIRE達成時点で厚生年金の加入を停止し、その時点までの加入期間で年金額を計算する。
    国民年金は20歳から60歳まで（最大40年）加入すると仮定。

    年金繰り下げ戦略が有効な場合、資産状況に応じて受給開始年齢を動的に決定する。

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        fire_achieved: FIRE達成フラグ
        fire_year_offset: FIRE達成時点の経過年数（年）
        current_assets: 現在の総資産（年金繰り下げ判定用）
        fire_target_assets: FIRE目標資産（年金繰り下げ判定用）
        override_start_ages: 年金開始年齢を直接指定する辞書（例: {'修平': 70, '桜': 65}）。
                             指定時は _determine_optimal_pension_start_age() をスキップ。

    Returns:
        年間年金収入（円）
    """
    if not _is_enabled(config, 'pension'):
        return 0

    pension_config = config['pension']
    default_start_age = pension_config['start_age']
    deferral_config = config['pension_deferral']

    # 現在の年齢を計算
    start_age_config = config['simulation']['start_age']
    current_age = start_age_config + year_offset

    # 年金成長率（マクロ経済スライド相当）
    pension_growth_rate = config['simulation']['standard']['pension_growth_rate']
    inflation_factor = (1 + pension_growth_rate) ** year_offset

    # override_start_ages 指定時: 各人の開始年齢を直接使用
    if override_start_ages is not None:
        increase_rate = deferral_config['deferral_increase_rate']
        decrease_rate = deferral_config['early_decrease_rate']

        total_pension = 0
        for person in pension_config['people']:
            person_name = person['name']
            person_start_age = override_start_ages.get(person_name, default_start_age)

            birthdate_str = person['birthdate']
            if birthdate_str:
                person_age = _get_age_at_offset(birthdate_str, year_offset)
                if person_age < person_start_age:
                    continue
            elif current_age < person_start_age:
                continue

            base = _calculate_person_pension(
                person, year_offset, person_start_age, fire_achieved, fire_year_offset
            )
            age_diff = person_start_age - default_start_age
            if age_diff > 0:
                adj = 1 + (increase_rate * age_diff)
            elif age_diff < 0:
                adj = 1 - (decrease_rate * abs(age_diff))
            else:
                adj = 1.0
            total_pension += base * adj
        return total_pension * inflation_factor

    # 繰り下げ戦略が無効、または資産情報がない場合は通常受給
    if not deferral_config['enabled'] or current_assets is None:
        if current_age < default_start_age:
            return 0

        base_pension = sum(
            _calculate_person_pension(person, year_offset, default_start_age, fire_achieved, fire_year_offset)
            for person in pension_config['people']
        )
        return base_pension * inflation_factor

    # 資産状況に基づいて最適な開始年齢を決定
    optimal_start_age = _determine_optimal_pension_start_age(
        current_assets=current_assets,
        config=config,
        fire_target_assets=fire_target_assets
    )

    min_start_age = deferral_config['min_start_age']

    if current_age < min_start_age:
        return 0

    if current_age < optimal_start_age:
        return 0

    base_pension = sum(
        _calculate_person_pension(person, year_offset, optimal_start_age, fire_achieved, fire_year_offset)
        for person in pension_config['people']
    )

    age_diff = optimal_start_age - default_start_age

    if age_diff > 0:
        increase_rate = deferral_config['deferral_increase_rate']
        adjustment_rate = 1 + (increase_rate * age_diff)
    elif age_diff < 0:
        decrease_rate = deferral_config['early_decrease_rate']
        adjustment_rate = 1 - (decrease_rate * abs(age_diff))
    else:
        adjustment_rate = 1.0

    return base_pension * adjustment_rate * inflation_factor


def calculate_child_allowance(year_offset: float, config: Dict[str, Any]) -> float:
    """
    指定年における児童手当を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書

    Returns:
        年間児童手当額（円）
    """
    # 児童手当が無効の場合は0を返す
    if not _is_enabled(config, 'child_allowance'):
        return 0

    children = config['education']['children']
    if not children:
        return 0

    allowance_config = config['child_allowance']
    # 2024年10月改定: 第2子以降0-3歳は2万円/月、支給対象を高校生（18歳未満）まで拡大
    first_child_under_3 = allowance_config['first_child_under_3']
    second_child_plus_under_3 = allowance_config['second_child_plus_under_3']
    age_3_to_high_school = allowance_config['age_3_to_high_school']

    total_annual_allowance = 0

    for i, child in enumerate(children):
        birthdate_str = child['birthdate']
        if not birthdate_str:
            continue

        child_age = _get_age_at_offset(birthdate_str, year_offset)

        # 年齢に応じた月額手当を計算（2024年10月改定後）
        if child_age < 3:
            # 第1子と第2子以降で金額が異なる
            monthly_allowance = first_child_under_3 if i == 0 else second_child_plus_under_3
        elif child_age < 18:
            # 3歳以上高校生以下（18歳未満）: 1万円/月
            monthly_allowance = age_3_to_high_school
        else:
            monthly_allowance = 0

        total_annual_allowance += monthly_allowance * 12

    return total_annual_allowance


def calculate_national_pension_premium(
    year_offset: float,
    config: Dict[str, Any],
    fire_achieved: bool = False
) -> float:
    """
    国民年金保険料を計算（FIRE後のみ計上）

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        fire_achieved: FIRE達成済みか（True: FIRE後、False: FIRE前）

    Returns:
        年間国民年金保険料（円）
    """
    # 社会保険設定が無効の場合は0を返す
    if not _is_enabled(config, 'social_insurance'):
        return 0

    # FIRE前は会社の厚生年金なので計上不要
    if not fire_achieved:
        return 0

    social_insurance_config = config['social_insurance']
    monthly_premium = social_insurance_config['national_pension_monthly_premium']

    # 年金受給者情報を取得
    pension_people = config['pension']['people']
    if not pension_people:
        return 0

    total_annual_premium = 0

    for person in pension_people:
        birthdate_str = person['birthdate']
        if not birthdate_str:
            continue

        # シミュレーション中の年齢を計算
        person_age = _get_age_at_offset(birthdate_str, year_offset)

        # 20歳～60歳の間のみ国民年金保険料を支払う
        if 20 <= person_age < 60:
            total_annual_premium += monthly_premium * 12

    return total_annual_premium


def calculate_national_health_insurance_premium(
    year_offset: float,
    config: Dict[str, Any],
    fire_achieved: bool = False,
    prev_year_capital_gains: float = 0,
    years_since_fire: float = 0,
) -> float:
    """
    国民健康保険料を動的計算（FIRE後のみ計上）

    国民健康保険料 = 所得割 + 均等割 + 平等割（上限あり）
    所得割は前年の所得（副業収入 + 株式譲渡益）に基づいて計算する。

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        fire_achieved: FIRE達成済みか（True: FIRE後、False: FIRE前）
        prev_year_capital_gains: 前年の株式譲渡益（円）

    Returns:
        年間国民健康保険料（円）
    """
    # 社会保険設定が無効の場合は0を返す
    if not _is_enabled(config, 'social_insurance'):
        return 0

    # FIRE前は会社の健康保険なので計上不要
    if not fire_achieved:
        return 0

    si = config['social_insurance']

    # --- 所得の算出 ---
    # 副業収入（修平 + 桜の年額合計）- FIRE後固定額
    annual_side_income = _compute_post_fire_income(config) * 12

    # 前年の株式譲渡益（キャピタルゲイン）
    # 国民健康保険の所得割は分離課税の譲渡所得も含む
    capital_gains = prev_year_capital_gains

    # 合計所得
    total_income = annual_side_income + capital_gains

    # --- 所得割 ---
    basic_deduction = si['health_insurance_basic_deduction']
    taxable_income = max(0, total_income - basic_deduction)
    income_rate = si['health_insurance_income_rate']
    income_based_premium = taxable_income * income_rate

    # --- 均等割 + 平等割 ---
    per_person = si['health_insurance_per_person']
    members = si['health_insurance_members']
    per_household = si['health_insurance_per_household']
    fixed_premium = per_person * members + per_household

    # --- 合計（上限適用）---
    max_premium = si['health_insurance_max_premium']
    total_premium = min(income_based_premium + fixed_premium, max_premium)

    return total_premium


def calculate_mortgage_payment(year_offset: float, config: Dict[str, Any]) -> float:
    """
    指定年における住宅ローンの月次支払額を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書

    Returns:
        月次住宅ローン支払額（円）
    """
    # 住宅ローンが無効の場合は0を返す
    if not _is_enabled(config, 'mortgage'):
        return 0

    mortgage_config = config['mortgage']
    monthly_payment = mortgage_config['monthly_payment']
    end_date_str = mortgage_config['end_date']

    if not end_date_str:
        return 0

    ref_date = _get_reference_date()
    end_date = _parse_birthdate(end_date_str)

    months_offset = int(year_offset * 12)
    simulation_date = ref_date + relativedelta(months=months_offset)

    # 終了日を過ぎていれば0、そうでなければ月次支払額を返す
    if simulation_date > end_date:
        return 0
    else:
        return monthly_payment


def calculate_house_maintenance(year_offset: float, config: Dict[str, Any]) -> float:
    """
    指定年における住宅メンテナンス費用を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書

    Returns:
        年間メンテナンス費用（円）
    """
    # 住宅メンテナンスが無効の場合は0を返す
    if not _is_enabled(config, 'house_maintenance'):
        return 0

    maintenance_config = config['house_maintenance']
    items = maintenance_config['items']

    if not items:
        return 0

    ref_date = _get_reference_date()
    current_year = ref_date.year

    simulation_year = int(current_year + year_offset)

    total_maintenance_cost = 0

    for item in items:
        first_year = item['first_year']
        frequency_years = item['frequency_years']
        cost = item['cost']

        if not first_year or not frequency_years:
            continue

        # 初回実施年以降かチェック
        if simulation_year < first_year:
            continue

        # 周期に該当するかチェック
        years_since_first = simulation_year - first_year
        if years_since_first % frequency_years == 0:
            total_maintenance_cost += cost

    return total_maintenance_cost


def calculate_workation_cost(year_offset: float, config: Dict[str, Any]) -> float:
    """
    指定年におけるワーケーション費用を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書

    Returns:
        年間ワーケーション費用（円）
    """
    # ワーケーションが無効の場合は0を返す
    if not _is_enabled(config, 'workation'):
        return 0

    workation_config = config['workation']
    start_child_index = workation_config['start_child_index']
    start_child_age = workation_config['start_child_age']
    annual_cost = workation_config['annual_cost']

    # 子供情報を取得
    children = config['education']['children']
    if start_child_index >= len(children):
        return 0  # 指定された子供が存在しない

    # 基準となる子供の情報
    child = children[start_child_index]
    birthdate_str = child['birthdate']
    if not birthdate_str:
        return 0

    # シミュレーション時点での子供の年齢を計算
    child_age = _get_age_at_offset(birthdate_str, year_offset)

    # 開始年齢以降ならワーケーション費用を返す
    if child_age >= start_child_age:
        return annual_cost
    else:
        return 0


def _sell_stocks_with_tax(
    shortage: float,
    stocks: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    stocks_cost_basis: float,
    capital_gains_tax_rate: float,
    allocation_enabled: bool,
) -> _StockSaleResult:
    """
    NISA優先で株を売却し、shortage分を確保する。

    呼び出し側での使い分け:
      支出不足型: cash += result.cash_from_taxable         （NISA分はcashを経由せず支出に充当）
      最低現金型: cash += result.nisa_sold + result.cash_from_taxable（全額cashへ）

    Returns:
      stocks, nisa_balance, nisa_cost_basis, stocks_cost_basis: 更新後の値
      nisa_sold:         NISA売却額（非課税）
      cash_from_taxable: 課税口座売却後の現金（税引後）
      capital_gain:      実現益（呼び出し側でcapital_gains_this_yearに加算）
      total_sold:        総売却額
    """
    nisa_sold = 0.0
    cash_from_taxable = 0.0
    capital_gain = 0.0

    if not allocation_enabled:
        # 資産配分無効: 税なし単純売却（全額cash_from_taxableとして返す）
        sold = min(shortage, stocks)
        return _StockSaleResult(
            stocks=stocks - sold,
            nisa_balance=nisa_balance,
            nisa_cost_basis=nisa_cost_basis,
            stocks_cost_basis=max(0.0, stocks_cost_basis - sold),
            nisa_sold=0.0,
            cash_from_taxable=sold,
            capital_gain=0.0,
            total_sold=sold,
        )

    # NISA優先売却（非課税）
    if nisa_balance > 0 and shortage > 0:
        nisa_sold = min(shortage, nisa_balance)
        nisa_sold_cost = nisa_sold / nisa_balance * nisa_cost_basis if nisa_balance > 0 else 0.0
        nisa_balance -= nisa_sold
        nisa_cost_basis = max(0.0, nisa_cost_basis - nisa_sold_cost)
        stocks -= nisa_sold
        stocks_cost_basis = max(0.0, stocks_cost_basis - nisa_sold_cost)
        shortage -= nisa_sold

    # 課税口座売却（税引後でshorageを確保）
    taxable_sold = 0.0
    if shortage > 0 and stocks > 0:
        taxable_stocks = stocks - nisa_balance
        if taxable_stocks > 0:
            avg_cost_basis = (stocks_cost_basis - nisa_cost_basis) / taxable_stocks
            gain_ratio = max(0.0, 1.0 - avg_cost_basis)
            effective_tax_rate = capital_gains_tax_rate * gain_ratio
            required_sale = shortage / (1 - effective_tax_rate) if effective_tax_rate < 1 else shortage
            taxable_sold = min(required_sale, taxable_stocks)
            sale_cost = taxable_sold * avg_cost_basis
            capital_gain = max(0.0, taxable_sold - sale_cost)
            tax = capital_gain * capital_gains_tax_rate
            cash_from_taxable = taxable_sold - tax
            stocks -= taxable_sold
            stocks_cost_basis = max(0.0, stocks_cost_basis - sale_cost)

    return _StockSaleResult(
        stocks=stocks,
        nisa_balance=nisa_balance,
        nisa_cost_basis=nisa_cost_basis,
        stocks_cost_basis=stocks_cost_basis,
        nisa_sold=nisa_sold,
        cash_from_taxable=cash_from_taxable,
        capital_gain=capital_gain,
        total_sold=nisa_sold + taxable_sold,
    )


def _calculate_monthly_expenses(
    years: float,
    config: Dict[str, Any],
    monthly_expense: float,
    expense_growth_rate: float,
    fire_achieved: bool,
    prev_year_capital_gains: float,
    years_since_fire: float = 0,
) -> dict:
    """
    月次支出の全項目を計算する。

    FIRE後は _compute_post_fire_monthly_expenses に委譲し、
    全シミュレーションパスで支出計算ロジックを共有する。

    Returns:
        base_expense, education_expense, mortgage_payment, maintenance_cost,
        workation_cost, pension_premium, health_insurance_premium, total（全て月額・円）
    """
    if fire_achieved:
        return _compute_post_fire_monthly_expenses(
            years, config,
            prev_year_capital_gains=prev_year_capital_gains,
            years_since_fire=years_since_fire,
        )

    # FIRE前: 成長率ベースのフォールバック値を使用
    fallback_annual = monthly_expense * 12 * (1 + expense_growth_rate) ** years
    base_expense = calculate_base_expense(years, config, fallback_annual, expense_growth_rate) / 12
    education_expense = calculate_education_expense(years, config) / 12
    mortgage_payment = calculate_mortgage_payment(years, config)
    maintenance_cost = calculate_house_maintenance(years, config) / 12
    workation_cost = calculate_workation_cost(years, config) / 12
    pension_premium = calculate_national_pension_premium(years, config, False) / 12
    health_insurance_premium = calculate_national_health_insurance_premium(
        years, config, False, prev_year_capital_gains,
        years_since_fire=years_since_fire,
    ) / 12
    total = (base_expense + education_expense + mortgage_payment
             + maintenance_cost + workation_cost
             + pension_premium + health_insurance_premium)
    return {
        'base_expense': base_expense,
        'education_expense': education_expense,
        'mortgage_payment': mortgage_payment,
        'maintenance_cost': maintenance_cost,
        'workation_cost': workation_cost,
        'pension_premium': pension_premium,
        'health_insurance_premium': health_insurance_premium,
        'total': total,
    }


def _auto_invest_surplus(
    cash: float,
    stocks: float,
    stocks_cost_basis: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    nisa_used_this_year: float,
    expense: float,
    cash_buffer_months: float,
    min_cash_balance: float,
    auto_invest_threshold: float,
    nisa_enabled: bool,
    nisa_annual_limit: float,
    invest_beyond_nisa: bool,
) -> dict:
    """
    余剰現金をNISA優先で自動投資する（FIRE前専用）。

    Returns:
        cash, stocks, stocks_cost_basis, nisa_balance,
        nisa_cost_basis, nisa_used_this_year, auto_invested
    """
    auto_invested = 0
    expense_based_buffer = expense * cash_buffer_months
    required_cash_balance = max(expense_based_buffer, min_cash_balance)
    cash_threshold = required_cash_balance * auto_invest_threshold

    if cash > cash_threshold:
        surplus = cash - required_cash_balance

        nisa_remaining = nisa_annual_limit - nisa_used_this_year if nisa_enabled else 0
        if nisa_remaining > 0 and surplus > 0:
            nisa_invest = min(surplus, nisa_remaining)
            cash -= nisa_invest
            stocks += nisa_invest
            stocks_cost_basis += nisa_invest
            nisa_balance += nisa_invest
            nisa_cost_basis += nisa_invest
            nisa_used_this_year += nisa_invest
            auto_invested += nisa_invest
            surplus -= nisa_invest

        if surplus > 0 and invest_beyond_nisa:
            taxable_invest = surplus
            cash -= taxable_invest
            stocks += taxable_invest
            stocks_cost_basis += taxable_invest
            auto_invested += taxable_invest

        if cash < min_cash_balance:
            adjustment = min_cash_balance - cash
            cash += adjustment
            stocks -= adjustment
            stocks_cost_basis -= adjustment
            auto_invested -= adjustment

    return {
        'cash': cash, 'stocks': stocks, 'stocks_cost_basis': stocks_cost_basis,
        'nisa_balance': nisa_balance, 'nisa_cost_basis': nisa_cost_basis,
        'nisa_used_this_year': nisa_used_this_year, 'auto_invested': auto_invested,
    }


def _apply_monthly_investment_returns(
    stocks: float,
    nisa_balance: float,
    monthly_return_rate: float
) -> Dict[str, float]:
    """
    月次運用リターンを適用（株式とNISA両方）
    
    重要: この関数は3つの月次処理関数から呼び出される共通ロジック。
    NISA残高への運用リターン適用漏れは過去のバグの原因だったため、
    この関数で一元管理する。
    
    Args:
        stocks: 現在の株式残高
        nisa_balance: 現在のNISA残高
        monthly_return_rate: 月次リターン率
    
    Returns:
        {
            'stocks': 更新後の株式残高,
            'nisa_balance': 更新後のNISA残高,
            'investment_return': 株式の運用リターン額
        }
    
    Raises:
        AssertionError: NISA残高が株式残高を超えた場合
    """
    investment_return = stocks * monthly_return_rate
    stocks += investment_return
    nisa_balance *= (1 + monthly_return_rate)
    
    # 不変条件チェック: NISA残高は株式残高を超えてはならない
    assert nisa_balance <= stocks + 1e-6, (
        f"NISA balance ({nisa_balance:,.0f}) cannot exceed total stocks ({stocks:,.0f}). "
        "This indicates a bug in NISA calculation logic."
    )
    
    return {
        'stocks': stocks,
        'nisa_balance': nisa_balance,
        'investment_return': investment_return
    }


def _maintain_minimum_cash_balance(
    cash: float,
    stocks: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    stocks_cost_basis: float,
    min_cash_balance: float,
    capital_gains_tax_rate: float,
    allocation_enabled: bool,
) -> Dict[str, Any]:
    """
    最低現金残高を維持（不足時に株式を売却）

    FIRE前専用: 固定の最低現金残高を維持する。
    FIRE後は別のロジック(_manage_post_fire_cash)を使用する。

    Args:
        cash: 現在の現金残高
        stocks: 現在の株式残高
        nisa_balance: 現在のNISA残高
        nisa_cost_basis: NISA簿価
        stocks_cost_basis: 株式簿価
        min_cash_balance: 最低現金残高
        capital_gains_tax_rate: 譲渡益税率
        allocation_enabled: 資産配分が有効か

    Returns:
        {
            'cash': 更新後の現金,
            'stocks': 更新後の株式,
            'nisa_balance': 更新後のNISA残高,
            'nisa_cost_basis': 更新後のNISA簿価,
            'stocks_cost_basis': 更新後の株式簿価,
            'capital_gain': 譲渡益
        }
    """
    capital_gain = 0

    if not allocation_enabled:
        return {
            'cash': cash,
            'stocks': stocks,
            'nisa_balance': nisa_balance,
            'nisa_cost_basis': nisa_cost_basis,
            'stocks_cost_basis': stocks_cost_basis,
            'capital_gain': capital_gain
        }

    # 現金不足時に株式を売却
    if cash < min_cash_balance and stocks > 0:
        shortage = min_cash_balance - cash
        result = _sell_stocks_with_tax(
            shortage, stocks, nisa_balance, nisa_cost_basis,
            stocks_cost_basis, capital_gains_tax_rate, allocation_enabled,
        )
        stocks = result.stocks
        nisa_balance = result.nisa_balance
        nisa_cost_basis = result.nisa_cost_basis
        stocks_cost_basis = result.stocks_cost_basis
        cash += result.nisa_sold + result.cash_from_taxable
        capital_gain = result.capital_gain

    return {
        'cash': cash,
        'stocks': stocks,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'stocks_cost_basis': stocks_cost_basis,
        'capital_gain': capital_gain
    }


def _manage_post_fire_cash(
    cash: float,
    stocks: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    stocks_cost_basis: float,
    monthly_expense: float,
    drawdown: float,
    config: Dict[str, Any],
    capital_gains_tax_rate: float,
    allocation_enabled: bool,
    is_start_of_month: bool,
) -> Dict[str, Any]:
    """
    FIRE後専用の現金管理戦略を実行する。
    
    平常時:
      - 現金確保目標(target_cash_reserve) + 生活費バッファを確保
      - 月初に不足分を株式から現金に変換
    
    暴落時:
      - 株式売却を停止（底値売却回避）
      - 確保済み現金から取り崩す
      - 復帰条件: 緊急時(emergency_cash_floor) or 回復(recovery_threshold)
    
    Args:
        cash: 現在の現金残高
        stocks: 現在の株式残高
        nisa_balance: 現在のNISA残高
        nisa_cost_basis: NISA簿価
        stocks_cost_basis: 株式簿価
        monthly_expense: 月次支出額
        drawdown: 現在のドローダウン（負の値）
        config: 設定辞書
        capital_gains_tax_rate: 譲渡益税率
        allocation_enabled: 資産配分が有効か
        is_start_of_month: 月初かどうか
    
    Returns:
        {
            'cash': 更新後の現金,
            'stocks': 更新後の株式,
            'nisa_balance': 更新後のNISA残高,
            'nisa_cost_basis': 更新後のNISA簿価,
            'stocks_cost_basis': 更新後の株式簿価,
            'capital_gain': 譲渡益,
            'in_market_crash': 暴落中かどうか,
            'stocks_sold_for_monthly': 月初の定期売却額
        }
    """
    capital_gain = 0
    stocks_sold_for_monthly = 0
    
    if not allocation_enabled:
        return {
            'cash': cash,
            'stocks': stocks,
            'nisa_balance': nisa_balance,
            'nisa_cost_basis': nisa_cost_basis,
            'stocks_cost_basis': stocks_cost_basis,
            'capital_gain': capital_gain,
            'in_market_crash': False,
            'stocks_sold_for_monthly': 0,
        }
    
    # 設定を取得
    strategy_config = config['post_fire_cash_strategy']
    if not strategy_config['enabled']:
        # 戦略が無効の場合は従来ロジック（何もしない）
        return {
            'cash': cash,
            'stocks': stocks,
            'nisa_balance': nisa_balance,
            'nisa_cost_basis': nisa_cost_basis,
            'stocks_cost_basis': stocks_cost_basis,
            'capital_gain': capital_gain,
            'in_market_crash': False,
            'stocks_sold_for_monthly': 0,
        }
    
    target_cash_reserve = strategy_config['target_cash_reserve']
    monthly_buffer_months = strategy_config['monthly_buffer_months']
    crash_threshold = strategy_config['market_crash_threshold']
    recovery_threshold = strategy_config['recovery_threshold']
    emergency_floor = strategy_config['emergency_cash_floor']
    
    # 暴落判定
    in_market_crash = drawdown <= crash_threshold
    is_recovering = drawdown >= recovery_threshold
    is_emergency = cash < emergency_floor

    # 目標現金レベル: 現金確保目標 + 生活費バッファ
    target_cash_level = target_cash_reserve + (monthly_buffer_months * monthly_expense)

    # 月初処理: 現金が目標レベルを下回っている場合のみ株式売却
    # （収入がある場合は、収入だけで足りる可能性があるため）
    if is_start_of_month:
        # 暴落中でも以下の条件で売却再開:
        # 1. 回復した（ドローダウン ≥ -10%）
        # 2. 緊急時（現金 < 25万円）
        should_sell_monthly = (not in_market_crash) or is_recovering or is_emergency

        # 現金が目標レベルを下回っている場合のみ売却
        cash_shortage = target_cash_level - cash
        if should_sell_monthly and cash_shortage > 0 and stocks > 0:
            result = _sell_stocks_with_tax(
                cash_shortage, stocks, nisa_balance, nisa_cost_basis,
                stocks_cost_basis, capital_gains_tax_rate, allocation_enabled,
            )
            stocks = result.stocks
            nisa_balance = result.nisa_balance
            nisa_cost_basis = result.nisa_cost_basis
            stocks_cost_basis = result.stocks_cost_basis
            cash += result.nisa_sold + result.cash_from_taxable
            capital_gain += result.capital_gain
            stocks_sold_for_monthly = result.total_sold
    
    return {
        'cash': cash,
        'stocks': stocks,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'stocks_cost_basis': stocks_cost_basis,
        'capital_gain': capital_gain,
        'in_market_crash': in_market_crash,
        'stocks_sold_for_monthly': stocks_sold_for_monthly,
    }


def _process_monthly_expense(
    cash: float,
    expense: float,
    stocks: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    stocks_cost_basis: float,
    capital_gains_tax_rate: float,
    allocation_enabled: bool
) -> Dict[str, Any]:
    """
    月次支出を処理（現金不足時は株式取り崩し）
    
    重要: この関数は2つの月次処理関数から呼び出される共通ロジック。
    支出処理の一貫性を保ち、修正漏れを防ぐために一元管理する。
    
    Args:
        cash: 現在の現金残高
        expense: 月次支出額
        stocks: 現在の株式残高
        nisa_balance: 現在のNISA残高
        nisa_cost_basis: NISA簿価
        stocks_cost_basis: 株式簿価
        capital_gains_tax_rate: 譲渡益税率
        allocation_enabled: 資産配分が有効か
    
    Returns:
        {
            'cash': 更新後の現金,
            'stocks': 更新後の株式,
            'nisa_balance': 更新後のNISA残高,
            'nisa_cost_basis': 更新後のNISA簿価,
            'stocks_cost_basis': 更新後の株式簿価,
            'capital_gain': 譲渡益,
            'withdrawal_from_stocks': 株式からの取り崩し額,
            'capital_gains_tax': 譲渡益税
        }
    """
    withdrawal_from_stocks = 0
    capital_gains_tax = 0
    capital_gain = 0
    
    if cash >= expense:
        cash -= expense
    else:
        shortage = expense - cash
        cash = 0
        
        result = _sell_stocks_with_tax(
            shortage, stocks, nisa_balance, nisa_cost_basis,
            stocks_cost_basis, capital_gains_tax_rate, allocation_enabled,
        )
        stocks = result.stocks
        nisa_balance = result.nisa_balance
        nisa_cost_basis = result.nisa_cost_basis
        stocks_cost_basis = result.stocks_cost_basis
        # NISA分は支出に直接充当、課税口座分は税引後現金として支出に充当
        # 株式売却で得た合計(nisa_sold + cash_from_taxable)から支出(shortage)を差し引いた残余のみ現金化
        total_proceeds = result.nisa_sold + result.cash_from_taxable
        cash += max(0, total_proceeds - shortage)
        capital_gain = result.capital_gain
        withdrawal_from_stocks = result.total_sold
        capital_gains_tax = result.capital_gain * capital_gains_tax_rate
    
    return {
        'cash': cash,
        'stocks': stocks,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'stocks_cost_basis': stocks_cost_basis,
        'capital_gain': capital_gain,
        'withdrawal_from_stocks': withdrawal_from_stocks,
        'capital_gains_tax': capital_gains_tax
    }


def _build_monthly_result(
    date, month: int,
    cash: float, stocks: float, stocks_cost_basis: float, nisa_balance: float,
    total_income: float, monthly_pension_income: float, labor_income: float,
    shuhei_income_monthly: float, sakura_income_monthly: float,
    monthly_child_allowance: float,
    expense: float, base_expense: float, monthly_education_expense: float,
    monthly_mortgage_payment: float, monthly_maintenance_cost: float,
    monthly_workation_cost: float, monthly_pension_premium: float,
    monthly_health_insurance_premium: float,
    investment_return: float, auto_invested: float,
    capital_gains_tax: float, fire_achieved: bool, fire_month,
) -> dict:
    """月次シミュレーション結果の1行分を構築する。"""
    return {
        'date': date, 'month': month,
        'assets': max(0, cash + stocks),
        'cash': max(0, cash), 'stocks': max(0, stocks),
        'stocks_cost_basis': max(0, stocks_cost_basis),
        'nisa_balance': max(0, nisa_balance),
        'income': total_income,
        'pension_income': monthly_pension_income,
        'labor_income': labor_income,
        'shuhei_income': shuhei_income_monthly,
        'sakura_income': sakura_income_monthly,
        'child_allowance': monthly_child_allowance,
        'expense': expense, 'base_expense': base_expense,
        'education_expense': monthly_education_expense,
        'mortgage_payment': monthly_mortgage_payment,
        'maintenance_cost': monthly_maintenance_cost,
        'workation_cost': monthly_workation_cost,
        'pension_premium': monthly_pension_premium,
        'health_insurance_premium': monthly_health_insurance_premium,
        'net_cashflow': total_income - expense,
        'investment_return': investment_return,
        'auto_invested': auto_invested,
        'capital_gains_tax': capital_gains_tax,
        'fire_achieved': fire_achieved,
        'fire_month': fire_month,
    }


def _sakura_income_for_month(date: datetime, sakura_income_base: float, config: dict) -> float:
    """
    指定月の桜の月収を返す。産休・育休期間中は config の monthly_income を使用する。

    Args:
        date: 判定する月の日付
        sakura_income_base: 通常時の桜の月収
        config: 設定辞書

    Returns:
        その月の桜の月収（円）
    """
    leave_list = config['simulation']['maternity_leave']
    children = config['education']['children']

    for leave in leave_list:
        child_name = leave['child']
        months_before = leave['months_before']
        months_after = leave['months_after']
        income_during_leave = leave['monthly_income']

        # 対象の子供の生年月日を検索
        birthdate_str = next(
            (c['birthdate'] for c in children if c['name'] == child_name),
            None,
        )
        if birthdate_str is None:
            continue

        birthdate = datetime.strptime(birthdate_str, '%Y/%m/%d')
        leave_start = birthdate - relativedelta(months=months_before)
        leave_end = birthdate + relativedelta(months=months_after)

        if leave_start <= date <= leave_end:
            return income_during_leave

    return sakura_income_base


def _shuhei_income_for_month(date: datetime, shuhei_income_grown: float, config: dict) -> float:
    """
    指定月の修平の月収を返す。
    育休期間中は育児休業給付金、時短勤務期間中は減額した給与を適用。

    Args:
        date: 判定する月の日付
        shuhei_income_grown: 成長率適用後の通常時の修平の月収
        config: 設定辞書

    Returns:
        その月の修平の月収（円）
    """
    sim = config['simulation']
    children = config['education']['children']

    # 育休判定（優先）
    leave_list = sim['shuhei_parental_leave']
    for leave in leave_list:
        child_name = leave['child']
        months_after = leave['months_after']
        income_first = leave['monthly_income']
        income_later = leave['monthly_income_after_180days']

        birthdate_str = next(
            (c['birthdate'] for c in children if c['name'] == child_name),
            None,
        )
        if birthdate_str is None:
            continue

        birthdate = datetime.strptime(birthdate_str, '%Y/%m/%d')
        leave_start = birthdate
        leave_end = birthdate + relativedelta(months=months_after)

        if leave_start <= date <= leave_end:
            boundary_180 = birthdate + relativedelta(months=6)
            if date < boundary_180:
                return income_first
            return income_later

    # 時短勤務判定
    reduced_list = sim['shuhei_reduced_hours']
    for rh in reduced_list:
        child_name = rh['child']
        start_months = rh['start_months_after']
        end_months = rh['end_months_after']
        ratio = rh['income_ratio']

        birthdate_str = next(
            (c['birthdate'] for c in children if c['name'] == child_name),
            None,
        )
        if birthdate_str is None:
            continue

        birthdate = datetime.strptime(birthdate_str, '%Y/%m/%d')
        rh_start = birthdate + relativedelta(months=start_months)
        rh_end = birthdate + relativedelta(months=end_months)

        if rh_start <= date <= rh_end:
            return shuhei_income_grown * ratio

    return shuhei_income_grown


def _compute_post_fire_income(config: Dict[str, Any]) -> float:
    """FIRE後の労働収入（固定額）を返す。

    年金受給開始までの間、設定された固定額の収入を得る。
    """
    sim = config['simulation']
    shuhei_base = sim['shuhei_post_fire_income']
    sakura_base = sim['sakura_post_fire_income']
    return shuhei_base + sakura_base


def _calculate_monthly_income(
    years: float,
    date: datetime,
    fire_achieved: bool,
    fire_month,
    shuhei_income_base: float,
    sakura_income_base: float,
    monthly_income: float,
    shuhei_ratio: float,
    income_growth_rate: float,
    config: dict,
    current_assets: float = None,
    override_start_ages: Dict[str, int] = None,
) -> dict:
    """
    月次収入を計算する。FIRE前後で労働収入の扱いを切り替える。

    Args:
        current_assets: 現在の総資産（年金繰り下げ判定用、Noneの場合はデフォルト年齢で受給）

    Returns:
      total_income:          月次合計収入
      labor_income:               労働収入（FIRE後は shuhei + sakura の post_fire 合計）
      pension_income:             月次年金収入
      child_allowance:            月次児童手当
      shuhei_income_monthly:      修平の月収（FIRE後は shuhei_post_fire_income）
      sakura_income_monthly:      桜の月収（FIRE後は sakura_post_fire_income）
      post_fire_income:           修平の FIRE後副収入設定値
    """

    if fire_achieved:
        fire_year_offset = (fire_month / 12) if fire_month is not None else None
        _inc = _compute_post_fire_monthly_income(
            years, config, fire_year_offset=fire_year_offset,
            current_assets=current_assets,
            override_start_ages=override_start_ages,
        )
        labor_income = _inc['post_fire_income']
        sim = config['simulation']
        if _inc['pension_income'] > 0:
            shuhei_pf = 0
            sakura_pf = 0
        else:
            shuhei_pf = sim['shuhei_post_fire_income']
            sakura_pf = sim['sakura_post_fire_income']
        return {
            'total_income': _inc['total'],
            'labor_income': labor_income,
            'pension_income': _inc['pension_income'],
            'child_allowance': _inc['child_allowance'],
            'shuhei_income_monthly': shuhei_pf,
            'sakura_income_monthly': sakura_pf,
            'post_fire_income': shuhei_pf,
        }

    # FIRE前: 年金収入の計算
    fire_year_offset = (fire_month / 12) if fire_month is not None else None
    base_expense_empty_nest = config['fire']['base_expense_by_stage']['empty_nest']
    fire_target_assets = base_expense_empty_nest / 0.04

    annual_pension_income = calculate_pension_income(
        years, config, fire_achieved=False, fire_year_offset=fire_year_offset,
        current_assets=current_assets,
        fire_target_assets=fire_target_assets,
        override_start_ages=override_start_ages
    )
    monthly_pension_income = annual_pension_income / 12
    monthly_child_allowance = calculate_child_allowance(years, config) / 12

    shuhei_post_fire_income = config['simulation']['shuhei_post_fire_income']
    sakura_post_fire_income = config['simulation']['sakura_post_fire_income']

    # FIRE前: 労働収入を成長率に応じて計算
    # 修平（会社員）: income_growth_rateを適用。育休期間中は給付金に置換
    # 桜（個人事業主）: 固定（成長なし）。産休・育休期間中は月収を変動させる
    sakura_income_current = _sakura_income_for_month(date, sakura_income_base, config)
    if shuhei_income_base + sakura_income_base > 0:
        shuhei_income_grown = shuhei_income_base * (1 + income_growth_rate) ** years
        shuhei_income_monthly = _shuhei_income_for_month(date, shuhei_income_grown, config)
        sakura_income_monthly = sakura_income_current
        income = shuhei_income_monthly + sakura_income_monthly
    else:
        income = monthly_income * (1 + income_growth_rate) ** years
        shuhei_income_monthly = income * shuhei_ratio
        sakura_income_monthly = income * (1 - shuhei_ratio)

    return {
        'total_income': income + monthly_pension_income + monthly_child_allowance,
        'labor_income': income,
        'pension_income': monthly_pension_income,
        'child_allowance': monthly_child_allowance,
        'shuhei_income_monthly': shuhei_income_monthly,
        'sakura_income_monthly': sakura_income_monthly,
        'post_fire_income': shuhei_post_fire_income,
    }


def _initialize_post_fire_simulation(
    current_cash: float,
    current_stocks: float,
    years_offset: float,
    config: Dict[str, Any],
    scenario: str,
    nisa_balance: float,
    nisa_cost_basis: float,
    stocks_cost_basis: Optional[float]
) -> Dict[str, Any]:
    """
    FIRE後シミュレーションの初期状態を設定

    Returns:
        初期化済みの設定と状態変数を含む辞書
    """
    # シナリオ設定取得
    scenario_config = config['simulation'][scenario]
    annual_return_rate = scenario_config['annual_return_rate']

    # シミュレーション期間
    life_expectancy = config['simulation']['life_expectancy']
    start_age = config['simulation']['start_age']
    current_age = start_age + years_offset
    remaining_years = life_expectancy - current_age
    remaining_months = int(remaining_years * 12)

    # 月次リターン率
    monthly_return_rate = _get_monthly_return_rate(annual_return_rate)

    # 資産配分設定
    allocation_enabled = _is_enabled(config, 'asset_allocation')
    if allocation_enabled:
        capital_gains_tax_rate = config['asset_allocation']['capital_gains_tax_rate']
        min_cash_balance = config['asset_allocation']['min_cash_balance']
    else:
        capital_gains_tax_rate = 0.20315
        min_cash_balance = 0

    # FIRE後の収入
    post_fire_income = (
        config['simulation']['shuhei_post_fire_income']
        + config['simulation']['sakura_post_fire_income']
    )

    return {
        'cash': current_cash,
        'stocks': current_stocks,
        'stocks_cost_basis': stocks_cost_basis if stocks_cost_basis is not None else current_stocks,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'remaining_months': remaining_months,
        'monthly_return_rate': monthly_return_rate,
        'allocation_enabled': allocation_enabled,
        'capital_gains_tax_rate': capital_gains_tax_rate,
        'min_cash_balance': min_cash_balance,
        'post_fire_income': post_fire_income,
    }


def _get_life_stage_for_year(years: float, config: Dict[str, Any]) -> str:
    """指定年数におけるライフステージ名を取得する。"""
    children_cfg = config['education']['children']
    _sub = config['fire']['empty_nest_sub_stages']
    _parent_age = _get_primary_parent_age(config, years)
    if children_cfg:
        child_age = _get_age_at_offset(children_cfg[0]['birthdate'], years)
        return _get_life_stage(child_age, _parent_age, _sub['senior_from_age'], _sub['elderly_from_age'])
    else:
        return _get_life_stage(22.0, _parent_age, _sub['senior_from_age'], _sub['elderly_from_age'])


def _get_discretionary_ratio(years: float, config: Dict[str, Any]) -> float:
    """現在のライフステージに応じた裁量的支出比率を取得する。

    緊縮削減と動的支出調整の両方で共通利用される。
    """
    stage = _get_life_stage_for_year(years, config)
    disc_ratios = config['fire']['discretionary_ratio_by_stage']
    return disc_ratios.get(stage, 0.30)


def _compute_post_fire_monthly_expenses(
    years: float,
    config: Dict[str, Any],
    prev_year_capital_gains: float = 0,
    years_since_fire: float = 0,
    include_health_insurance: bool = True,
    extra_monthly_budget: float = 0,
) -> dict:
    """FIRE後の月次支出を計算する単一関数。

    全シミュレーションパス（ベースライン、MC事前計算、MCイテレーション、FIRE判定）が
    この関数を利用し、支出計算ロジックの重複を排除する。

    Args:
        include_health_insurance: Falseの場合、健康保険料を0とする（MC事前計算用）

    Returns:
        各項目の月額 + total + annual_base_expense を含む辞書
    """
    annual_base_expense = calculate_base_expense(years, config, 0)
    base_expense = annual_base_expense / 12
    education_expense = calculate_education_expense(years, config) / 12
    mortgage_payment = calculate_mortgage_payment(years, config)
    maintenance_cost = calculate_house_maintenance(years, config) / 12
    workation_cost = calculate_workation_cost(years, config) / 12
    pension_premium = calculate_national_pension_premium(years, config, fire_achieved=True) / 12

    health_insurance_premium = 0.0
    if include_health_insurance:
        health_insurance_premium = calculate_national_health_insurance_premium(
            years, config, fire_achieved=True,
            prev_year_capital_gains=prev_year_capital_gains,
            years_since_fire=years_since_fire,
        ) / 12

    total = (base_expense + education_expense + mortgage_payment
             + maintenance_cost + workation_cost
             + pension_premium + health_insurance_premium
             + extra_monthly_budget)

    return {
        'base_expense': base_expense,
        'education_expense': education_expense,
        'mortgage_payment': mortgage_payment,
        'maintenance_cost': maintenance_cost,
        'workation_cost': workation_cost,
        'pension_premium': pension_premium,
        'health_insurance_premium': health_insurance_premium,
        'total': total,
        'annual_base_expense': annual_base_expense,
    }


def _compute_post_fire_monthly_income(
    years: float,
    config: Dict[str, Any],
    fire_year_offset: float,
    current_assets: float = None,
    override_start_ages: Dict[str, int] = None,
    post_fire_income_override: float = None,
) -> dict:
    """FIRE後の月次収入を計算する単一関数。

    全シミュレーションパスがこの関数を利用し、収入計算ロジックの重複を排除する。

    Args:
        current_assets: 現在の総資産（年金繰り下げ判定用、Noneの場合は override_start_ages で判定）
        post_fire_income_override: 労働収入の上書き値（Noneの場合はconfigから取得）

    Returns:
        pension_income, child_allowance, post_fire_income, total を含む辞書
    """
    base_expense_empty_nest = config['fire']['base_expense_by_stage']['empty_nest']
    fire_target_assets = base_expense_empty_nest / 0.04

    annual_pension_income = calculate_pension_income(
        years, config, fire_achieved=True, fire_year_offset=fire_year_offset,
        current_assets=current_assets,
        fire_target_assets=fire_target_assets,
        override_start_ages=override_start_ages,
    )
    monthly_pension_income = annual_pension_income / 12

    monthly_child_allowance = calculate_child_allowance(years, config) / 12

    if monthly_pension_income > 0:
        labor_income = 0.0
    elif post_fire_income_override is not None:
        labor_income = post_fire_income_override
    else:
        labor_income = _compute_post_fire_income(config)

    total = monthly_pension_income + monthly_child_allowance + labor_income

    return {
        'pension_income': monthly_pension_income,
        'child_allowance': monthly_child_allowance,
        'post_fire_income': labor_income,
        'total': total,
    }


def _compute_austerity_period(
    config: Dict[str, Any],
    fire_years_offset: float,
    override_start_ages: Dict[str, int] = None,
) -> Tuple[Optional[int], Optional[int], float]:
    """緊縮期間の開始月・終了月・削減率を計算する。

    ループ外で1回だけ呼び、結果を _compute_austerity_reduction に渡す。

    Returns:
        (reduction_start_month, reduction_end_month, reduction_rate)
        適用不要の場合は (None, None, 0.0)
    """
    opt_cfg = config.get('optimization', {})
    red_rate = opt_cfg.get('pre_pension_reduction_rate') or 0.0
    if red_rate <= 0:
        return None, None, 0.0

    austerity_years = opt_cfg.get('austerity_years_before_pension', 10)
    austerity_months = austerity_years * 12

    pension_people = config['pension']['people']
    ages_list = []
    for p in pension_people:
        ovr = p.get('override_start_age')
        if ovr:
            ages_list.append(ovr)
        elif override_start_ages and p['name'] in override_start_ages:
            ages_list.append(override_start_ages[p['name']])

    if not ages_list:
        return None, None, 0.0

    earliest_pension_age = min(ages_list)
    start_age = config['simulation']['start_age']
    reduction_end_month = int((earliest_pension_age - start_age - fire_years_offset) * 12)
    reduction_start_month = max(0, reduction_end_month - austerity_months)

    return reduction_start_month, reduction_end_month, red_rate


def _compute_austerity_reduction(
    months_since_fire: int,
    years: float,
    config: Dict[str, Any],
    reduction_rate: float,
    reduction_start_month: int,
    reduction_end_month: int,
) -> float:
    """緊縮期間中の裁量的支出削減額を計算する。

    Returns:
        削減額（正の値）。緊縮期間外は 0.0。
    """
    if reduction_rate <= 0:
        return 0.0
    if reduction_start_month is None or reduction_end_month is None:
        return 0.0
    if not (reduction_start_month <= months_since_fire < reduction_end_month):
        return 0.0

    disc_ratio = _get_discretionary_ratio(years, config)
    annual_base = calculate_base_expense(years, config, 0)
    disc_monthly = annual_base * disc_ratio / 12.0
    return disc_monthly * reduction_rate


def _process_post_fire_monthly_cycle(
    month: int,
    cash: float,
    stocks: float,
    stocks_cost_basis: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    current_year_post: int,
    capital_gains_this_year_post: float,
    prev_year_capital_gains_post: float,
    years_offset: float,
    config: Dict[str, Any],
    current_date_post: datetime,
    monthly_return_rate: float,
    allocation_enabled: bool,
    capital_gains_tax_rate: float,
    min_cash_balance: float,
    post_fire_income: float,
    override_start_ages: Dict[str, int] = None,
    extra_monthly_budget: float = 0,
    austerity_start: int = None,
    austerity_end: int = None,
    austerity_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    FIRE後の1ヶ月分のシミュレーション処理（決定論的ベースライン用）

    Returns:
        更新後の状態と破綻フラグを含む辞書
    """
    years = years_offset + month / 12
    date_post = current_date_post + relativedelta(months=int(years * 12))

    # 年度管理
    _prev_gains = capital_gains_this_year_post
    current_year_post, capital_gains_this_year_post, _year_advanced = _advance_year(
        date_post.year, current_year_post, capital_gains_this_year_post
    )
    if _year_advanced:
        prev_year_capital_gains_post = _prev_gains

    current_total_assets = cash + stocks
    drawdown = 0.0

    # 支出計算（共通関数）
    _exp = _compute_post_fire_monthly_expenses(
        years, config,
        prev_year_capital_gains=prev_year_capital_gains_post,
        years_since_fire=month / 12,
        extra_monthly_budget=extra_monthly_budget,
    )
    expense = _exp['total']

    # 緊縮削減
    expense -= _compute_austerity_reduction(
        month, years, config, austerity_rate, austerity_start, austerity_end,
    )

    # 収入計算（共通関数）
    _inc = _compute_post_fire_monthly_income(
        years, config, fire_year_offset=years_offset,
        current_assets=current_total_assets,
        override_start_ages=override_start_ages,
    )
    total_income = _inc['total']

    cash += total_income

    # FIRE後の現金管理戦略（支出処理の前に実行）
    if allocation_enabled:
        cash_mgmt_result = _manage_post_fire_cash(
            cash, stocks, nisa_balance, nisa_cost_basis, stocks_cost_basis,
            expense, drawdown, config,
            capital_gains_tax_rate, allocation_enabled, True
        )
        cash = cash_mgmt_result['cash']
        stocks = cash_mgmt_result['stocks']
        nisa_balance = cash_mgmt_result['nisa_balance']
        nisa_cost_basis = cash_mgmt_result['nisa_cost_basis']
        stocks_cost_basis = cash_mgmt_result['stocks_cost_basis']
        capital_gains_this_year_post += cash_mgmt_result['capital_gain']

    expense_result = _process_monthly_expense(
        cash, expense, stocks, nisa_balance, nisa_cost_basis,
        stocks_cost_basis, capital_gains_tax_rate, allocation_enabled
    )
    cash = expense_result['cash']
    stocks = expense_result['stocks']
    nisa_balance = expense_result['nisa_balance']
    nisa_cost_basis = expense_result['nisa_cost_basis']
    stocks_cost_basis = expense_result['stocks_cost_basis']
    capital_gains_this_year_post += expense_result['capital_gain']

    returns_result = _apply_monthly_investment_returns(
        stocks, nisa_balance, monthly_return_rate
    )
    stocks = returns_result['stocks']
    nisa_balance = returns_result['nisa_balance']
    investment_return = returns_result['investment_return']

    safety_margin = config['post_fire_cash_strategy']['safety_margin']
    should_break = (cash + stocks <= safety_margin)

    return {
        'cash': cash,
        'stocks': stocks,
        'stocks_cost_basis': stocks_cost_basis,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'current_year_post': current_year_post,
        'capital_gains_this_year_post': capital_gains_this_year_post,
        'prev_year_capital_gains_post': prev_year_capital_gains_post,
        'should_break': should_break,
    }


def simulate_post_fire_assets(
    current_cash: float,
    current_stocks: float,
    years_offset: float,
    config: Dict[str, Any],
    scenario: str = 'standard',
    nisa_balance: float = 0,
    nisa_cost_basis: float = 0,
    stocks_cost_basis: float = None,
    override_start_ages: Dict[str, int] = None,
    extra_monthly_budget: float = 0,
) -> float:
    """
    FIRE後の資産推移をシミュレーション（実際のロジックと同じ）

    Args:
        current_cash: 現在の現金残高
        current_stocks: 現在の株式残高
        years_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        scenario: シナリオ名
        nisa_balance: NISA残高
        nisa_cost_basis: NISA簿価
        stocks_cost_basis: 株式全体の簿価（Noneの場合は時価と仮定）

    Returns:
        90歳時点での資産額
    """
    if _REFERENCE_DATE is None:
        _set_reference_date()

    init = _initialize_post_fire_simulation(
        current_cash, current_stocks, years_offset, config, scenario,
        nisa_balance, nisa_cost_basis, stocks_cost_basis
    )

    cash = init['cash']
    stocks = init['stocks']
    stocks_cost_basis = init['stocks_cost_basis']
    nisa_balance = init['nisa_balance']
    nisa_cost_basis = init['nisa_cost_basis']
    remaining_months = init['remaining_months']
    monthly_return_rate = init['monthly_return_rate']
    allocation_enabled = init['allocation_enabled']
    capital_gains_tax_rate = init['capital_gains_tax_rate']
    min_cash_balance = init['min_cash_balance']
    post_fire_income = init['post_fire_income']

    current_date_post = _get_reference_date()
    current_year_post = (current_date_post + relativedelta(months=int(years_offset * 12))).year
    capital_gains_this_year_post = 0
    prev_year_capital_gains_post = 0

    # 緊縮期間の事前計算
    a_start, a_end, a_rate = _compute_austerity_period(
        config, years_offset, override_start_ages
    )

    # 月次シミュレーション
    for month in range(remaining_months):
        _post_fire_income = _compute_post_fire_income(config)
        cycle_result = _process_post_fire_monthly_cycle(
            month, cash, stocks, stocks_cost_basis, nisa_balance, nisa_cost_basis,
            current_year_post, capital_gains_this_year_post, prev_year_capital_gains_post,
            years_offset, config, current_date_post,
            monthly_return_rate, allocation_enabled,
            capital_gains_tax_rate, min_cash_balance, _post_fire_income,
            override_start_ages=override_start_ages,
            extra_monthly_budget=extra_monthly_budget,
            austerity_start=a_start,
            austerity_end=a_end,
            austerity_rate=a_rate,
        )

        # 状態を更新
        cash = cycle_result['cash']
        stocks = cycle_result['stocks']
        stocks_cost_basis = cycle_result['stocks_cost_basis']
        nisa_balance = cycle_result['nisa_balance']
        nisa_cost_basis = cycle_result['nisa_cost_basis']
        current_year_post = cycle_result['current_year_post']
        capital_gains_this_year_post = cycle_result['capital_gains_this_year_post']
        prev_year_capital_gains_post = cycle_result['prev_year_capital_gains_post']

        # 破綻判定
        if cycle_result['should_break']:
            return 0

    return cash + stocks

def _precompute_monthly_cashflows(
    years_offset: float,
    total_months: int,
    config: Dict[str, Any],
    post_fire_income: float = 0,  # 後方互換性のため残す（内部では使用しない）
    override_start_ages: Dict[str, int] = None,
    post_fire_income_override: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    モンテカルロ用に月次支出・収入を事前計算

    Args:
        years_offset: FIRE達成時の経過年数
        total_months: 計算する月数
        config: 設定辞書
        post_fire_income: FIRE後の労働収入

    Returns:
        (expenses_array, income_array, base_expenses_array, life_stages_array, workation_costs_array):
            - expenses_array: 各月の総支出（基本生活費+教育費+住宅ローン等）
            - income_array: 各月の収入
            - base_expenses_array: 各月の基本生活費（年額）
            - life_stages_array: 各月のライフステージ（文字列）
            - workation_costs_array: 各月のワーケーション費用（月額）※動的削減の対象
    """
    expenses = np.zeros(total_months)
    income = np.zeros(total_months)
    base_expenses = np.zeros(total_months)
    life_stages = np.empty(total_months, dtype=object)
    workation_costs = np.zeros(total_months)

    for month_idx in range(total_months):
        years = years_offset + month_idx / 12

        # 支出計算（共通関数、健康保険は確率的譲渡益に依存するため除外）
        _exp = _compute_post_fire_monthly_expenses(
            years, config, include_health_insurance=False,
        )
        expenses[month_idx] = _exp['total']
        base_expenses[month_idx] = _exp['annual_base_expense']
        workation_costs[month_idx] = _exp['workation_cost']

        # ライフステージを取得して記録（動的調整用）
        life_stages[month_idx] = _get_life_stage_for_year(years, config)

        # 収入計算（共通関数、current_assets=None: 事前計算時は資産額未確定）
        _inc = _compute_post_fire_monthly_income(
            years, config, fire_year_offset=years_offset,
            current_assets=None,
            override_start_ages=override_start_ages,
            post_fire_income_override=post_fire_income_override,
        )
        income[month_idx] = _inc['total']

    return expenses, income, base_expenses, life_stages, workation_costs


def _simulate_post_fire_with_random_returns(
    current_cash: float,
    current_stocks: float,
    years_offset: float,
    config: Dict[str, Any],
    scenario: str,
    random_returns: np.ndarray,
    nisa_balance: float = 0,
    nisa_cost_basis: float = 0,
    stocks_cost_basis: float = None,
    return_timeseries: bool = False,
    precomputed_expenses: np.ndarray = None,
    precomputed_income: np.ndarray = None,
    precomputed_base_expenses: np.ndarray = None,
    precomputed_life_stages: np.ndarray = None,
    precomputed_workation_costs: np.ndarray = None,
    baseline_assets: np.ndarray = None,
    override_start_ages: Dict[str, int] = None,
    extra_monthly_budget: float = 0,
    return_min_path_assets: bool = False,
    pre_pension_reduction_rate: float = 0.0,
    reduction_end_month: int = None,
    austerity_months: int = 120,
):
    """
    FIRE後の資産推移をランダムリターンでシミュレーション（モンテカルロ用）

    Args:
        current_cash: FIRE達成時の現金残高
        current_stocks: FIRE達成時の株式残高
        years_offset: FIRE達成時のシミュレーション開始からの経過年数
        config: 設定辞書
        scenario: シナリオ名
        random_returns: 月次リターン率の配列
        nisa_balance: NISA残高
        nisa_cost_basis: NISA簿価
        stocks_cost_basis: 株式全体の簿価
        return_timeseries: Trueの場合、月ごとの資産リストを返す
        return_min_path_assets: Trueの場合、(final_assets, min_path_assets) を返す
        pre_pension_reduction_rate: 年金受給前の緊縮期間における裁量的支出の削減率 (0-1)
        reduction_end_month: 緊縮期間の終了月（年金受給開始月）
        austerity_months: 緊縮期間の長さ（月数、デフォルト120=10年）

    Returns:
        return_timeseries=True: 月ごとの資産額のリスト
        return_min_path_assets=True: (final_assets, min_path_assets) のタプル
        それ以外: 最終資産額（破綻時は0）
    """
    # 初期化
    init = _initialize_post_fire_simulation(
        current_cash, current_stocks, years_offset, config, scenario,
        nisa_balance, nisa_cost_basis, stocks_cost_basis
    )

    cash = init['cash']
    stocks = init['stocks']
    stocks_cost_basis = init['stocks_cost_basis']
    nisa_balance = init['nisa_balance']
    nisa_cost_basis = init['nisa_cost_basis']
    remaining_months = init['remaining_months']
    allocation_enabled = init['allocation_enabled']
    capital_gains_tax_rate = init['capital_gains_tax_rate']
    min_cash_balance = init['min_cash_balance']
    post_fire_income = init['post_fire_income']

    current_date_post = _get_reference_date()
    current_year_post = (current_date_post + relativedelta(months=int(years_offset * 12))).year
    capital_gains_this_year_post = 0
    prev_year_capital_gains_post = 0

    # ランダムリターンの長さチェック
    if len(random_returns) < remaining_months:
        raise ValueError(f"random_returns length ({len(random_returns)}) < remaining_months ({remaining_months})")

    # 月ごとの資産を記録（return_timeseries=Trueの場合）
    timeseries = [] if return_timeseries else None
    _last_positive_baseline_scalar = float(baseline_assets[0]) if baseline_assets is not None and len(baseline_assets) > 0 and baseline_assets[0] > 0 else 0.0

    min_path_assets = current_cash + current_stocks

    reduction_start_month = None
    if pre_pension_reduction_rate > 0 and reduction_end_month is not None:
        reduction_start_month = max(0, reduction_end_month - austerity_months)

    # 月次シミュレーション
    for month in range(remaining_months):
        years = years_offset + month / 12
        date_post = current_date_post + relativedelta(months=int(years * 12))

        # 年度管理
        _prev_gains = capital_gains_this_year_post
        current_year_post, capital_gains_this_year_post, _year_advanced = _advance_year(
            date_post.year, current_year_post, capital_gains_this_year_post
        )
        if _year_advanced:
            prev_year_capital_gains_post = _prev_gains

        # 計画資産比ドローダウン計算（ベースラインがない場合は削減なし）
        current_total_assets = cash + stocks
        _ba_val_scalar = 0.0
        if baseline_assets is not None and month < len(baseline_assets):
            _ba_val_scalar = float(baseline_assets[month])
            if _ba_val_scalar > 0:
                _last_positive_baseline_scalar = _ba_val_scalar
        _planned = _last_positive_baseline_scalar if _last_positive_baseline_scalar > 0 else None
        drawdown, _ = calculate_drawdown_level(
            current_assets=current_total_assets,
            peak_assets_history=[],
            config=config,
            planned_assets=_planned,
        )

        # 支出・収入計算（事前計算済みの場合は配列から取得）
        if precomputed_expenses is not None and precomputed_income is not None:
            base_expense_total_original = precomputed_expenses[month]

            annual_health_insurance_premium = calculate_national_health_insurance_premium(
                years, config, fire_achieved=True,
                prev_year_capital_gains=prev_year_capital_gains_post,
                years_since_fire=month / 12,
            )
            monthly_health_insurance_premium = annual_health_insurance_premium / 12

            expense = base_expense_total_original + monthly_health_insurance_premium + extra_monthly_budget
            total_income = precomputed_income[month]

            # 比例制御型動的支出調整（プリコンピュートパス）
            if baseline_assets is not None and _last_positive_baseline_scalar > 0:
                surplus = current_total_assets - _last_positive_baseline_scalar
                if precomputed_base_expenses is not None and precomputed_life_stages is not None:
                    stage = precomputed_life_stages[month]
                    disc_ratios = config['fire']['discretionary_ratio_by_stage']
                    disc_ratio = disc_ratios.get(stage, 0.30)
                else:
                    disc_ratio = _get_discretionary_ratio(years, config)
                disc_monthly = precomputed_base_expenses[month] * disc_ratio / 12.0 if precomputed_base_expenses is not None else expense * 0.30
                adjustment = calculate_proportional_expense_adjustment(surplus, disc_monthly, config)
                expense += adjustment
        else:
            # インラインパス（共通関数使用）
            _exp = _compute_post_fire_monthly_expenses(
                years, config,
                prev_year_capital_gains=prev_year_capital_gains_post,
                years_since_fire=month / 12,
                extra_monthly_budget=extra_monthly_budget,
            )
            expense = _exp['total']

            # 比例制御型動的支出調整（インラインパス）
            if baseline_assets is not None and _last_positive_baseline_scalar > 0:
                surplus = current_total_assets - _last_positive_baseline_scalar
                disc_ratio = _get_discretionary_ratio(years, config)
                disc_monthly = _exp['annual_base_expense'] * disc_ratio / 12.0
                adjustment = calculate_proportional_expense_adjustment(surplus, disc_monthly, config)
                expense += adjustment

            # 収入計算（共通関数使用）
            _inc = _compute_post_fire_monthly_income(
                years, config, fire_year_offset=years_offset,
                current_assets=current_total_assets,
                override_start_ages=override_start_ages,
            )
            total_income = _inc['total']

        # 緊縮期間の支出削減（共通関数使用）
        expense -= _compute_austerity_reduction(
            month, years, config,
            pre_pension_reduction_rate, reduction_start_month, reduction_end_month,
        )

        # 収入を現金に加算
        cash += total_income

        # FIRE後の現金管理戦略を適用（支出処理の前）
        if allocation_enabled:
            # 月次支出を推定（必要に応じて計算）
            if precomputed_expenses is not None:
                estimated_monthly_expense = precomputed_expenses[month]
            else:
                estimated_monthly_expense = expense

            # 月初判定（常にtrue - モンテカルロでは毎月処理）
            is_start_of_month = True

            # FIRE後の現金管理を実行
            cash_result = _manage_post_fire_cash(
                cash, stocks, nisa_balance, nisa_cost_basis, stocks_cost_basis,
                estimated_monthly_expense, drawdown, config,
                capital_gains_tax_rate, allocation_enabled, is_start_of_month
            )
            cash = cash_result['cash']
            stocks = cash_result['stocks']
            nisa_balance = cash_result['nisa_balance']
            nisa_cost_basis = cash_result['nisa_cost_basis']
            stocks_cost_basis = cash_result['stocks_cost_basis']
            capital_gains_this_year_post += cash_result['capital_gain']

        # 支出を現金から引き出し
        if cash >= expense:
            cash -= expense
        else:
            shortage = expense - cash
            cash = 0
            result = _sell_stocks_with_tax(
                shortage, stocks, nisa_balance, nisa_cost_basis,
                stocks_cost_basis, capital_gains_tax_rate, allocation_enabled,
            )
            stocks = result.stocks
            nisa_balance = result.nisa_balance
            nisa_cost_basis = result.nisa_cost_basis
            stocks_cost_basis = result.stocks_cost_basis
            cash += result.cash_from_taxable
            capital_gains_this_year_post += result.capital_gain

        # ランダムリターン適用（NISA含む）
        monthly_return = random_returns[month]
        stocks *= (1 + monthly_return)
        nisa_balance *= (1 + monthly_return)

        total_assets_now = cash + stocks

        # 月ごとの資産を記録
        if return_timeseries:
            timeseries.append(total_assets_now)

        min_path_assets = min(min_path_assets, total_assets_now)

        safety_margin = config['post_fire_cash_strategy']['safety_margin']
        if total_assets_now <= safety_margin:
            if return_timeseries:
                timeseries.extend([0] * (remaining_months - month - 1))
                return timeseries
            elif return_min_path_assets:
                return (0, min_path_assets)
            else:
                return 0

    final_assets = cash + stocks
    if return_timeseries:
        return timeseries
    elif return_min_path_assets:
        return (final_assets, min_path_assets)
    else:
        return final_assets



def can_retire_now(
    current_assets: float,
    years_offset: float,
    current_annual_expense: float,
    config: Dict[str, Any],
    scenario: str,
    current_cash: float = None,
    current_stocks: float = None,
    nisa_balance: float = 0,
    nisa_cost_basis: float = 0,
    stocks_cost_basis: float = None,
    override_start_ages: Dict[str, int] = None
) -> bool:
    """
    現在のタイミングで退職して、寿命まで資産が持つかチェック

    実際のシミュレーションと同じロジックを使用

    Args:
        current_assets: 現在の資産（cash+stocksが未指定時のみ使用）
        years_offset: シミュレーション開始からの経過年数
        current_annual_expense: 現在の年間支出
        config: 設定辞書
        scenario: シナリオ名
        current_cash: 現在の現金残高（オプション）
        current_stocks: 現在の株式残高（オプション）
        nisa_balance: NISA残高（オプション）
        nisa_cost_basis: NISA簿価（オプション）
        stocks_cost_basis: 株式全体の簿価（オプション）

    Returns:
        True if 退職可能、False otherwise
    """
    life_expectancy = config['simulation']['life_expectancy']
    start_age = config['simulation']['start_age']

    # 現在の年齢
    current_age = start_age + years_offset

    # 残り年数
    remaining_years = life_expectancy - current_age

    if remaining_years <= 0:
        return True  # すでに寿命到達

    # 現金と株の分離（指定されていない場合は全額株と仮定）
    if current_cash is None or current_stocks is None:
        # 資産配分が有効な場合は、適切に分離
        allocation_enabled = _is_enabled(config, 'asset_allocation')
        if allocation_enabled:
            cash_buffer_months = config['asset_allocation']['cash_buffer_months']
            monthly_expense = current_annual_expense / 12
            estimated_cash = monthly_expense * cash_buffer_months
            current_cash = min(estimated_cash, current_assets * 0.1)  # 最大10%を現金
            current_stocks = current_assets - current_cash
        else:
            current_cash = 0
            current_stocks = current_assets

    # 退職後のシミュレーション実行（実際のロジックと同じ）
    final_assets = simulate_post_fire_assets(
        current_cash=current_cash,
        current_stocks=current_stocks,
        years_offset=years_offset,
        config=config,
        scenario=scenario,
        nisa_balance=nisa_balance,
        nisa_cost_basis=nisa_cost_basis,
        stocks_cost_basis=stocks_cost_basis,
        override_start_ages=override_start_ages
    )

    return final_assets > config['post_fire_cash_strategy']['safety_margin']


def calculate_base_expense_by_category(
    year_offset: float,
    config: Dict[str, Any],
    fallback_expense: float
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    カテゴリ別予算に基づく基本生活費を計算

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        fallback_expense: フォールバック年間支出（使用しないが互換性のため保持）

    Returns:
        (total_expense, breakdown):
            - total_expense: 年間基本生活費合計（円）、無効の場合はNone
            - breakdown: カテゴリ別内訳辞書 {
                'categories': {cat_id: amount, ...},
                'essential_total': 基礎生活費合計,
                'discretionary_total': 裁量的支出合計,
                'stage': ライフステージ,
                'total': 合計
              }
    """
    # カテゴリ別予算が無効の場合はNoneを返す
    expense_categories = config['fire']['expense_categories']
    if not expense_categories['enabled']:
        return None, {}

    # カテゴリ定義と予算を取得
    definitions = expense_categories['definitions']
    budgets_by_stage = expense_categories['budgets_by_stage']

    if not definitions or not budgets_by_stage:
        return None, {}

    # カテゴリIDと裁量フラグのマップを作成
    discretionary_map = {cat['id']: cat['discretionary'] for cat in definitions}

    # 子供の情報を取得（最初の子供を基準にライフステージを決定）
    _sub = config['fire']['empty_nest_sub_stages']
    _senior_from = _sub['senior_from_age']
    _elderly_from = _sub['elderly_from_age']
    _parent_age = _get_primary_parent_age(config, year_offset)
    children = config['education']['children']
    if not children:
        stage = _get_life_stage(22.0, _parent_age, _senior_from, _elderly_from)
    else:
        # 最初の子供の年齢を計算してライフステージを決定
        child = children[0]
        birthdate_str = child['birthdate']
        if not birthdate_str:
            return None, {}

        child_age = _get_age_at_offset(birthdate_str, year_offset)
        if child_age < 0:
            stage = _get_life_stage(22.0, _parent_age, _senior_from, _elderly_from)
        else:
            stage = _get_life_stage(child_age, _parent_age, _senior_from, _elderly_from)

    # 該当ステージの予算を取得（empty_nest_* が未定義の場合は empty_nest にフォールバック）
    if stage not in budgets_by_stage:
        if stage.startswith('empty_nest_') and 'empty_nest' in budgets_by_stage:
            stage = 'empty_nest'
        else:
            return None, {}

    stage_budget = budgets_by_stage[stage]

    # カテゴリごとの金額を集計
    categories = {}
    essential_total = 0.0
    discretionary_total = 0.0

    for cat_id, amount in stage_budget.items():
        categories[cat_id] = amount
        if discretionary_map.get(cat_id, False):
            discretionary_total += amount
        else:
            essential_total += amount

    base_expense = essential_total + discretionary_total

    # 第二子以降：各子供の年齢（ステージ）に応じた追加費用を加算
    additional_by_stage = config['fire']['additional_child_expense_by_stage']

    if additional_by_stage and len(children) > 1:
        ref_date = _get_reference_date()
        simulation_date = ref_date + pd.Timedelta(days=year_offset * 365.25)

        for additional_child in children[1:]:
            child_birthdate_str = additional_child['birthdate']
            if not child_birthdate_str:
                continue

            child_birthdate = datetime.strptime(child_birthdate_str, '%Y/%m/%d')

            # まだ生まれていない場合はスキップ
            if child_birthdate > simulation_date:
                continue

            # 各子供の年齢からステージを判定
            child_age = _get_age_at_offset(child_birthdate_str, year_offset)
            child_stage = _get_life_stage(child_age)

            additional_expense = additional_by_stage.get(child_stage, 0)
            base_expense += additional_expense
            essential_total += additional_expense  # 追加費用は基礎生活費とみなす

    # 内訳辞書を作成
    breakdown = {
        'categories': categories,
        'essential_total': essential_total,
        'discretionary_total': discretionary_total,
        'stage': stage,
        'total': base_expense
    }

    return base_expense, breakdown


def calculate_base_expense(
    year_offset: float,
    config: Dict[str, Any],
    fallback_expense: float,
    expense_growth_rate: float = None,
) -> float:
    """
    指定年における基本生活費を計算（ライフステージ別 + 家族人数調整 + インフレ）

    カテゴリ別予算が有効な場合はそちらを優先、無効の場合は従来方式を使用。
    いずれの場合も expense_growth_rate による複利インフレを適用する。

    fallback_expense は呼び出し元でインフレ適用済みの値を受け取る前提。
    ステージ別・カテゴリ別の値はインフレ未適用の名目値のため、本関数内で適用する。

    Args:
        year_offset: シミュレーション開始からの経過年数
        config: 設定辞書
        fallback_expense: フォールバック年間支出（呼び出し元でインフレ適用済み）
        expense_growth_rate: インフレ率（Noneの場合config.simulation.standardから取得）

    Returns:
        年間基本生活費（円、インフレ適用済み）
    """
    if expense_growth_rate is None:
        expense_growth_rate = config['simulation']['standard']['expense_growth_rate']
    inflation_factor = (1 + expense_growth_rate) ** year_offset

    # カテゴリ別予算を試す
    category_expense, breakdown = calculate_base_expense_by_category(
        year_offset, config, fallback_expense
    )
    if category_expense is not None:
        return category_expense * inflation_factor

    # 従来方式（既存ロジック）
    # ============================
    manual_expense = config['fire']['manual_annual_expense']
    if manual_expense is not None:
        return manual_expense * inflation_factor

    base_expense_by_stage = config['fire']['base_expense_by_stage']

    if not base_expense_by_stage:
        return fallback_expense

    _sub = config['fire']['empty_nest_sub_stages']
    _senior_from = _sub['senior_from_age']
    _elderly_from = _sub['elderly_from_age']
    _parent_age = _get_primary_parent_age(config, year_offset)
    children = config['education']['children']

    def _resolve_empty_nest_stage() -> str:
        return _get_life_stage(22.0, _parent_age, _senior_from, _elderly_from)

    def _get_stage_expense(s: str) -> Optional[float]:
        if s in base_expense_by_stage:
            return base_expense_by_stage[s]
        if s.startswith('empty_nest_') and 'empty_nest' in base_expense_by_stage:
            return base_expense_by_stage['empty_nest']
        return None

    if not children:
        val = _get_stage_expense(_resolve_empty_nest_stage())
        return val * inflation_factor if val is not None else fallback_expense

    child = children[0]
    birthdate_str = child['birthdate']
    if not birthdate_str:
        return fallback_expense

    child_age = _get_age_at_offset(birthdate_str, year_offset)

    # 第一子が未出生の場合、子供がいないものとして扱う（Fix 5）
    if child_age < 0:
        val = _get_stage_expense(_resolve_empty_nest_stage())
        return val * inflation_factor if val is not None else fallback_expense

    stage = _get_life_stage(child_age, _parent_age, _senior_from, _elderly_from)

    # ステージキーが存在しない場合はfallback（既にインフレ適用済み）を返す
    val = _get_stage_expense(stage)
    if val is None:
        return fallback_expense
    base_expense = val

    additional_by_stage = config['fire']['additional_child_expense_by_stage']

    if additional_by_stage and len(children) > 1:
        ref_date = _get_reference_date()
        simulation_date = ref_date + pd.Timedelta(days=year_offset * 365.25)

        for additional_child in children[1:]:
            child_birthdate_str = additional_child['birthdate']
            if not child_birthdate_str:
                continue

            child_birthdate = datetime.strptime(child_birthdate_str, '%Y/%m/%d')

            if child_birthdate > simulation_date:
                continue

            child_age = _get_age_at_offset(child_birthdate_str, year_offset)
            if child_age < 0:
                continue
            child_stage = _get_life_stage(child_age)

            base_expense += additional_by_stage.get(child_stage, 0)

    return base_expense * inflation_factor


def _initialize_future_simulation(
    current_cash: Optional[float],
    current_stocks: Optional[float],
    current_assets: Optional[float],
    monthly_income: float,
    monthly_expense: float,
    config: Dict[str, Any],
    scenario: str
) -> Dict[str, Any]:
    """
    将来シミュレーションの初期状態を設定

    Returns:
        初期化済みの設定と状態変数を含む辞書
    """
    # 後方互換性: current_assetsが指定されている場合は全額株として扱う
    if current_cash is None and current_stocks is None:
        if current_assets is not None:
            cash = 0.0
            stocks = current_assets
        else:
            raise ValueError("Either (current_cash, current_stocks) or current_assets must be provided")
    else:
        cash = current_cash if current_cash is not None else 0.0
        stocks = current_stocks if current_stocks is not None else 0.0

    # シナリオ設定取得
    scenario_config = config['simulation'][scenario]
    annual_return_rate = scenario_config['annual_return_rate']
    inflation_rate = scenario_config['inflation_rate']
    income_growth_rate = scenario_config['income_growth_rate']
    expense_growth_rate = scenario_config['expense_growth_rate']

    # シミュレーション期間（寿命まで）
    life_expectancy = config['simulation']['life_expectancy']
    start_age = config['simulation']['start_age']
    simulation_years = life_expectancy - start_age
    simulation_months = simulation_years * 12

    # 月次リターン率（複利計算）
    monthly_return_rate = _get_monthly_return_rate(annual_return_rate)

    # 資産配分設定
    allocation_enabled = _is_enabled(config, 'asset_allocation')
    if allocation_enabled:
        cash_buffer_months = config['asset_allocation']['cash_buffer_months']
        auto_invest_threshold = config['asset_allocation']['auto_invest_threshold']
        nisa_enabled = config['asset_allocation']['nisa_enabled']
        nisa_annual_limit = config['asset_allocation']['nisa_annual_limit']
        invest_beyond_nisa = config['asset_allocation']['invest_beyond_nisa']
        min_cash_balance = config['asset_allocation']['min_cash_balance']
        capital_gains_tax_rate = config['asset_allocation']['capital_gains_tax_rate']
    else:
        cash_buffer_months = 0
        auto_invest_threshold = 999
        nisa_enabled = False
        nisa_annual_limit = 0
        invest_beyond_nisa = False
        min_cash_balance = 0
        capital_gains_tax_rate = 0.20315

    # 夫婦別収入の比率を計算
    shuhei_income_base = config['simulation']['shuhei_income']
    sakura_income_base = config['simulation']['sakura_income']
    if shuhei_income_base + sakura_income_base > 0:
        shuhei_ratio = shuhei_income_base / (shuhei_income_base + sakura_income_base)
    else:
        shuhei_ratio = 1.0

    return {
        # 資産
        'cash': cash,
        'stocks': stocks,
        'stocks_cost_basis': stocks,  # 初期の株は簿価=時価と仮定
        'nisa_balance': 0.0,
        'nisa_cost_basis': 0.0,

        # 設定
        'simulation_months': simulation_months,
        'monthly_return_rate': monthly_return_rate,
        'allocation_enabled': allocation_enabled,
        'cash_buffer_months': cash_buffer_months,
        'auto_invest_threshold': auto_invest_threshold,
        'nisa_enabled': nisa_enabled,
        'nisa_annual_limit': nisa_annual_limit,
        'invest_beyond_nisa': invest_beyond_nisa,
        'min_cash_balance': min_cash_balance,
        'capital_gains_tax_rate': capital_gains_tax_rate,

        # シナリオパラメータ
        'annual_return_rate': annual_return_rate,
        'inflation_rate': inflation_rate,
        'income_growth_rate': income_growth_rate,
        'expense_growth_rate': expense_growth_rate,

        # その他
        'shuhei_ratio': shuhei_ratio,
        'shuhei_income_base': shuhei_income_base,
        'sakura_income_base': sakura_income_base,
        'income': monthly_income,
        'expense': monthly_expense,
    }


def _process_future_monthly_cycle(
    month: int,
    cash: float,
    stocks: float,
    stocks_cost_basis: float,
    nisa_balance: float,
    nisa_cost_basis: float,
    fire_achieved: bool,
    fire_month: Optional[int],
    current_date: datetime,
    current_year: int,
    nisa_used_this_year: float,
    capital_gains_this_year: float,
    prev_year_capital_gains: float,
    config: Dict[str, Any],
    scenario: str,
    shuhei_income_base: float,
    sakura_income_base: float,
    shuhei_ratio: float,
    income: float,
    expense: float,
    monthly_return_rate: float,
    income_growth_rate: float,
    expense_growth_rate: float,
    allocation_enabled: bool,
    cash_buffer_months: float,
    min_cash_balance: float,
    auto_invest_threshold: float,
    nisa_enabled: bool,
    nisa_annual_limit: float,
    invest_beyond_nisa: bool,
    capital_gains_tax_rate: float,
    disable_fire_check: bool = False,
    override_start_ages: Dict[str, int] = None,
    force_fire_month: Optional[int] = None,
    extra_monthly_budget: float = 0,
) -> Dict[str, Any]:
    """
    1ヶ月分のシミュレーション処理

    Returns:
        {
            'cash': 更新後の現金,
            'stocks': 更新後の株式,
            'stocks_cost_basis': 更新後の株式簿価,
            'nisa_balance': 更新後のNISA残高,
            'nisa_cost_basis': 更新後のNISA簿価,
            'fire_achieved': FIRE達成フラグ,
            'fire_month': FIRE達成月,
            'current_year': 現在年度,
            'nisa_used_this_year': 今年のNISA投資額,
            'capital_gains_this_year': 今年の譲渡益,
            'prev_year_capital_gains': 前年の譲渡益,
            'monthly_result': 月次結果（DataFrame用）,
            'should_break': ループを終了すべきか
        }
    """
    # 現在月の日付
    date = current_date + relativedelta(months=month)

    # 年数（成長率計算用）
    years = month / 12

    # 年が変わったらNISA枠と譲渡益集計をリセット
    _prev_gains = capital_gains_this_year
    current_year, capital_gains_this_year, _year_advanced = _advance_year(
        date.year, current_year, capital_gains_this_year
    )
    if _year_advanced:
        nisa_used_this_year = 0
        prev_year_capital_gains = _prev_gains

    # 収入計算（年金繰り下げ戦略対応）
    current_total_assets = cash + stocks
    _income = _calculate_monthly_income(
        years, date, fire_achieved, fire_month,
        shuhei_income_base, sakura_income_base, income,
        shuhei_ratio, income_growth_rate, config,
        current_assets=current_total_assets,
        override_start_ages=override_start_ages,
    )
    total_income = _income['total_income']
    labor_income = _income['labor_income']
    monthly_pension_income = _income['pension_income']
    monthly_child_allowance = _income['child_allowance']
    shuhei_income_monthly = _income['shuhei_income_monthly']
    sakura_income_monthly = _income['sakura_income_monthly']

    # 支出計算
    _years_since_fire_fut = max(0.0, years - fire_month / 12) if fire_achieved and fire_month is not None else 0.0
    _exp = _calculate_monthly_expenses(
        years, config, expense, expense_growth_rate,
        fire_achieved, prev_year_capital_gains,
        years_since_fire=_years_since_fire_fut,
    )
    base_expense = _exp['base_expense']
    monthly_education_expense = _exp['education_expense']
    monthly_mortgage_payment = _exp['mortgage_payment']
    monthly_maintenance_cost = _exp['maintenance_cost']
    monthly_workation_cost = _exp['workation_cost']
    monthly_pension_premium = _exp['pension_premium']
    monthly_health_insurance_premium = _exp['health_insurance_premium']
    expense = _exp['total']

    if fire_achieved and extra_monthly_budget > 0:
        expense += extra_monthly_budget

    # 緊縮期間の支出削減（共通関数使用）
    if fire_achieved and fire_month is not None:
        _a_start, _a_end, _a_rate = _compute_austerity_period(
            config, fire_month / 12, override_start_ages,
        )
        expense -= _compute_austerity_reduction(
            month - fire_month, years, config, _a_rate, _a_start, _a_end,
        )

    # 1. 収入を現金に加算
    cash += total_income

    # 1.5. FIRE後は月初に現金管理戦略を適用（支出処理の前）
    # 重要: 支出処理の前に実行することで、二重売却を回避する
    # - このステップで生活費1ヶ月分を株式から現金に変換
    # - 次のステップでその現金から支出を引き出し
    if fire_achieved:
        # ベースラインシミュレーション（決定論的）ではドローダウン計算を行わない。
        # 動的支出削減はMCシミュレーション内で各パスのドローダウンに応じて適用される。
        # ベースラインはMC結果と比較するための楽観的基準線として機能する。
        drawdown = 0
        is_start_of_month = True  # ベースラインシミュレーションでは毎月処理

        cash_result = _manage_post_fire_cash(
            cash, stocks, nisa_balance, nisa_cost_basis, stocks_cost_basis,
            expense, drawdown, config, capital_gains_tax_rate, allocation_enabled,
            is_start_of_month
        )
        cash = cash_result['cash']
        stocks = cash_result['stocks']
        nisa_balance = cash_result['nisa_balance']
        nisa_cost_basis = cash_result['nisa_cost_basis']
        stocks_cost_basis = cash_result['stocks_cost_basis']
        capital_gains_this_year += cash_result['capital_gain']

    # 2. 支出を現金から引き出し
    # FIRE後: 上記で現金バッファを確保済みのため、通常は株式売却不要
    # FIRE前: 現金不足時に株式を売却して支出を賄う
    expense_result = _process_monthly_expense(
        cash, expense, stocks, nisa_balance, nisa_cost_basis,
        stocks_cost_basis, capital_gains_tax_rate, allocation_enabled
    )
    cash = expense_result['cash']
    stocks = expense_result['stocks']
    nisa_balance = expense_result['nisa_balance']
    nisa_cost_basis = expense_result['nisa_cost_basis']
    stocks_cost_basis = expense_result['stocks_cost_basis']
    capital_gains_this_year += expense_result['capital_gain']
    withdrawal_from_stocks = expense_result['withdrawal_from_stocks']
    capital_gains_tax = expense_result['capital_gains_tax']

    # 3. 運用リターン（株式とNISA両方に適用）
    returns_result = _apply_monthly_investment_returns(
        stocks, nisa_balance, monthly_return_rate
    )
    stocks = returns_result['stocks']
    nisa_balance = returns_result['nisa_balance']
    investment_return = returns_result['investment_return']

    # 4. 余剰現金がある場合は自動投資（FIRE前のみ）
    auto_invested = 0
    if allocation_enabled and not fire_achieved:
        _inv = _auto_invest_surplus(
            cash, stocks, stocks_cost_basis, nisa_balance, nisa_cost_basis,
            nisa_used_this_year, expense, cash_buffer_months, min_cash_balance,
            auto_invest_threshold, nisa_enabled, nisa_annual_limit, invest_beyond_nisa,
        )
        cash = _inv['cash']
        stocks = _inv['stocks']
        stocks_cost_basis = _inv['stocks_cost_basis']
        nisa_balance = _inv['nisa_balance']
        nisa_cost_basis = _inv['nisa_cost_basis']
        nisa_used_this_year = _inv['nisa_used_this_year']
        auto_invested = _inv['auto_invested']

        # 不変条件チェック: 自動投資後もNISA残高は株式残高以下
        assert nisa_balance <= stocks + 1e-6, (
            f"After auto-invest: NISA balance ({nisa_balance:,.0f}) cannot exceed stocks ({stocks:,.0f})"
        )

    # FIRE達成チェック
    total_assets = cash + stocks
    if not fire_achieved and month > 0 and not disable_fire_check:
        if force_fire_month is not None:
            if month >= force_fire_month:
                fire_achieved = True
                fire_month = month
                print(f"  FIRE強制達成 at month {month} ({years:.1f} years), assets=JPY{total_assets:,.0f}")
        else:
            annual_pension_premium_for_fire = calculate_national_pension_premium(years, config, fire_achieved=True)
            annual_health_insurance_premium_for_fire = calculate_national_health_insurance_premium(
                years, config, fire_achieved=True, prev_year_capital_gains=prev_year_capital_gains
            )
            current_annual_expense = (base_expense + monthly_education_expense + monthly_mortgage_payment + monthly_maintenance_cost + monthly_workation_cost) * 12 + annual_pension_premium_for_fire + annual_health_insurance_premium_for_fire

            if allocation_enabled:
                fire_cash_buffer = max(expense * cash_buffer_months, min_cash_balance)
            else:
                fire_cash_buffer = cash
            potential_investment = max(0, cash - fire_cash_buffer)

            fire_check_result = can_retire_now(
                current_assets=total_assets,
                years_offset=years,
                current_annual_expense=current_annual_expense,
                config=config,
                scenario=scenario,
                current_cash=cash,
                current_stocks=stocks,
                nisa_balance=nisa_balance,
                nisa_cost_basis=nisa_cost_basis,
                stocks_cost_basis=stocks_cost_basis,
                override_start_ages=override_start_ages
            )

            if fire_check_result:
                fire_achieved = True
                fire_month = month
                print(f"  FIRE可能! at month {month} ({years:.1f} years), assets=JPY{total_assets:,.0f} (cash={cash:,.0f}, stocks={stocks:,.0f}, potential_investment={potential_investment:,.0f})")

    # 記録
    monthly_result = _build_monthly_result(
        date, month, cash, stocks, stocks_cost_basis, nisa_balance,
        total_income, monthly_pension_income, labor_income,
        shuhei_income_monthly, sakura_income_monthly, monthly_child_allowance,
        expense, base_expense, monthly_education_expense,
        monthly_mortgage_payment, monthly_maintenance_cost, monthly_workation_cost,
        monthly_pension_premium, monthly_health_insurance_premium,
        investment_return, auto_invested, capital_gains_tax, fire_achieved, fire_month,
    )

    safety_margin = config['post_fire_cash_strategy']['safety_margin']
    should_break = (cash + stocks <= safety_margin)

    return {
        'cash': cash,
        'stocks': stocks,
        'stocks_cost_basis': stocks_cost_basis,
        'nisa_balance': nisa_balance,
        'nisa_cost_basis': nisa_cost_basis,
        'fire_achieved': fire_achieved,
        'fire_month': fire_month,
        'current_year': current_year,
        'nisa_used_this_year': nisa_used_this_year,
        'capital_gains_this_year': capital_gains_this_year,
        'prev_year_capital_gains': prev_year_capital_gains,
        'monthly_result': monthly_result,
        'should_break': should_break,
    }


def simulate_future_assets(
    current_cash: float = None,
    current_stocks: float = None,
    current_assets: float = None,  # 後方互換性のため
    monthly_income: float = 0,
    monthly_expense: float = 0,
    config: Dict[str, Any] = None,
    scenario: str = 'standard',
    disable_fire_check: bool = False,
    override_start_ages: Dict[str, int] = None,
    force_fire_month: Optional[int] = None,
    extra_monthly_budget: float = 0,
) -> pd.DataFrame:
    """
    将来の資産推移をシミュレーション（現金と株を分離管理）

    毎月「今退職しても寿命まで資産が持つか?」をチェックし、
    初めて持つと判定された月がFIRE達成日となる

    Args:
        current_cash: 現在の現金残高（優先）
        current_stocks: 現在の株式残高（優先）
        current_assets: 現在の純資産（後方互換性のため、cash/stocksが未指定時のみ使用）
        monthly_income: 月次収入
        monthly_expense: 月次支出
        config: 設定辞書
        scenario: シナリオ名（デフォルト: 'standard'）
        disable_fire_check: Trueの場合、FIRE判定をスキップし全期間を労働継続として実行

    Returns:
        シミュレーション結果のデータフレーム
    """
    print(f"Simulating future assets ({scenario} scenario)...")

    # 初期化処理
    init = _initialize_future_simulation(
        current_cash, current_stocks, current_assets,
        monthly_income, monthly_expense, config, scenario
    )

    # 初期化結果から変数を取得
    cash = init['cash']
    stocks = init['stocks']
    stocks_cost_basis = init['stocks_cost_basis']
    nisa_balance = init['nisa_balance']
    nisa_cost_basis = init['nisa_cost_basis']

    simulation_months = init['simulation_months']
    monthly_return_rate = init['monthly_return_rate']
    allocation_enabled = init['allocation_enabled']
    cash_buffer_months = init['cash_buffer_months']
    auto_invest_threshold = init['auto_invest_threshold']
    nisa_enabled = init['nisa_enabled']
    nisa_annual_limit = init['nisa_annual_limit']
    invest_beyond_nisa = init['invest_beyond_nisa']
    min_cash_balance = init['min_cash_balance']
    capital_gains_tax_rate = init['capital_gains_tax_rate']

    annual_return_rate = init['annual_return_rate']
    income_growth_rate = init['income_growth_rate']
    expense_growth_rate = init['expense_growth_rate']

    shuhei_ratio = init['shuhei_ratio']
    shuhei_income_base = init['shuhei_income_base']
    sakura_income_base = init['sakura_income_base']
    income = init['income']
    expense = init['expense']

    # シミュレーション結果を格納
    results = []

    # FIRE達成状態
    fire_achieved = False
    fire_month = None

    _set_reference_date()
    current_date = _get_reference_date()
    current_year = current_date.year
    nisa_used_this_year = 0
    capital_gains_this_year = 0
    prev_year_capital_gains = 0

    # 月次シミュレーション
    for month in range(simulation_months + 1):
        cycle_result = _process_future_monthly_cycle(
            month,
            cash, stocks, stocks_cost_basis, nisa_balance, nisa_cost_basis,
            fire_achieved, fire_month,
            current_date, current_year, nisa_used_this_year,
            capital_gains_this_year, prev_year_capital_gains,
            config, scenario,
            shuhei_income_base, sakura_income_base, shuhei_ratio,
            income, expense,
            monthly_return_rate, income_growth_rate, expense_growth_rate,
            allocation_enabled, cash_buffer_months, min_cash_balance,
            auto_invest_threshold, nisa_enabled, nisa_annual_limit,
            invest_beyond_nisa, capital_gains_tax_rate,
            disable_fire_check=disable_fire_check,
            override_start_ages=override_start_ages,
            force_fire_month=force_fire_month,
            extra_monthly_budget=extra_monthly_budget,
        )

        # 状態を更新
        cash = cycle_result['cash']
        stocks = cycle_result['stocks']
        stocks_cost_basis = cycle_result['stocks_cost_basis']
        nisa_balance = cycle_result['nisa_balance']
        nisa_cost_basis = cycle_result['nisa_cost_basis']
        fire_achieved = cycle_result['fire_achieved']
        fire_month = cycle_result['fire_month']
        current_year = cycle_result['current_year']
        nisa_used_this_year = cycle_result['nisa_used_this_year']
        capital_gains_this_year = cycle_result['capital_gains_this_year']
        prev_year_capital_gains = cycle_result['prev_year_capital_gains']

        # 結果を記録
        results.append(cycle_result['monthly_result'])

        # 破綻判定で終了
        if cycle_result['should_break']:
            break

    df = pd.DataFrame(results)

    return df


def _apply_baseline_floor(baseline_assets: list, initial_total: float, remaining_months: int) -> list:
    """ベースラインに線形償却フロアを適用する。

    ベースラインが破綻（0以下）した場合、線形スケジュール
    initial_total * (remaining - month) / remaining を最低参照値として使用し、
    ドローダウン検知が途切れることを防ぐ。
    """
    if not baseline_assets or initial_total <= 0 or remaining_months <= 0:
        return baseline_assets
    result = list(baseline_assets)
    for m in range(len(result)):
        linear_floor = initial_total * max(0, remaining_months - m) / remaining_months
        if result[m] < linear_floor:
            result[m] = linear_floor
    return result


def run_monte_carlo_simulation(
    current_cash: float,
    current_stocks: float,
    config: Dict[str, Any],
    scenario: str = 'standard',
    iterations: int = 1000,
    monthly_income: float = 0,
    monthly_expense: float = 0,
    override_start_ages: Dict[str, int] = None,
    min_fire_month: int = None,
    extra_monthly_budget: float = 0,
    post_fire_income_override: float = None,
) -> Dict[str, Any]:
    """
    モンテカルロシミュレーションを実行し、FIRE成功確率を計算

    Option 1実装: FIRE達成まで固定リターン、FIRE後のみランダムリターン
    - ベースシミュレーションでFIRE達成時点の状態を取得
    - その時点からFIRE後の期間のみランダムリターンでシミュレート
    - 「計画通りFIRE達成後の成功確率」を評価

    Args:
        current_cash: 現在の現金
        current_stocks: 現在の株式
        config: 設定辞書
        scenario: シナリオ名
        iterations: シミュレーション回数
        monthly_income: 月次労働収入
        monthly_expense: 月次支出
        override_start_ages: 年金開始年齢のオーバーライド
        min_fire_month: 指定するとこの月以降でFIREしたものとしてMCを実行

    Returns:
        {
            'success_rate': 成功確率（0-1）,
            'median_final_assets': 最終資産の中央値,
            'percentile_10': 下位10%の最終資産,
            'percentile_90': 上位10%の最終資産,
            'mean_final_assets': 最終資産の平均値,
            'all_results': 全イテレーションの結果リスト
        }
    """
    _set_reference_date()
    print(f"Running Monte Carlo simulation ({iterations} iterations)...")

    # ステップ1: ベースシミュレーション（固定リターン）でFIRE達成時点を取得
    print("  Step 1: Running base simulation to find FIRE achievement point...")

    if min_fire_month is not None:
        base_df = simulate_future_assets(
            current_cash=current_cash,
            current_stocks=current_stocks,
            config=config,
            scenario=scenario,
            monthly_income=monthly_income,
            monthly_expense=monthly_expense,
            override_start_ages=override_start_ages,
            disable_fire_check=True,
        )
        if min_fire_month >= len(base_df):
            raise ValueError(
                f"min_fire_month ({min_fire_month}) exceeds simulation length ({len(base_df)})"
            )
        fire_row = base_df.iloc[min_fire_month]
        fire_month = min_fire_month
    else:
        base_df = simulate_future_assets(
            current_cash=current_cash,
            current_stocks=current_stocks,
            config=config,
            scenario=scenario,
            monthly_income=monthly_income,
            monthly_expense=monthly_expense,
            override_start_ages=override_start_ages,
        )
        fire_rows = base_df[base_df['fire_achieved'] == True]
        if len(fire_rows) == 0:
            raise ValueError("FIRE not achieved in base simulation. Cannot run Monte Carlo.")
        fire_row = fire_rows.iloc[0]
        fire_month = int(fire_row['fire_month'])

    fire_cash = fire_row['cash']
    fire_stocks = fire_row['stocks']
    fire_nisa = fire_row['nisa_balance']
    fire_nisa_cost = fire_row.get('nisa_cost_basis', fire_nisa)
    fire_stocks_cost = fire_row['stocks_cost_basis']
    years_offset = fire_month / 12

    print(f"  FIRE achieved at month {fire_month} (age {config['simulation']['start_age'] + years_offset:.1f})")
    print(f"  Assets at FIRE: JPY{fire_cash + fire_stocks:,.0f}")

    # ステップ2: FIRE後の期間を計算
    life_expectancy = config['simulation']['life_expectancy']
    start_age = config['simulation']['start_age']
    fire_age = start_age + years_offset
    remaining_years = life_expectancy - fire_age
    remaining_months = int(remaining_years * 12)

    print(f"  Simulating {remaining_months} months post-FIRE with random returns...")

    # ステップ3: 支出・収入の事前計算（全イテレーションで共通）
    if post_fire_income_override is not None:
        post_fire_income = post_fire_income_override
    else:
        post_fire_income = (
            config['simulation']['shuhei_post_fire_income']
            + config['simulation']['sakura_post_fire_income']
        )

    # override_start_ages が未指定の場合:
    # 1. config に override_start_age が設定されていればそれを使用（決定論的と一致）
    # 2. なければ FIRE 時点の資産比率から年金開始年齢を決定する
    # （_precompute_monthly_cashflows は current_assets=None のため繰り下げ判定が行われず、
    #   全員65歳受給になってしまう。決定論的シミュレーションと一致させるために補正する。）
    if override_start_ages is None:
        config_override_ages = _extract_override_start_ages(config)
        if config_override_ages is not None:
            override_start_ages = config_override_ages
            print(f"  Pension start ages (from config): {override_start_ages}")
        else:
            optimal_pension_age = _determine_optimal_pension_start_age(
                current_assets=fire_cash + fire_stocks,
                config=config,
            )
            pension_people = config['pension']['people']
            override_start_ages = {p['name']: optimal_pension_age for p in pension_people}
            print(f"  Pension start age (derived from FIRE assets): {optimal_pension_age}")

    precomputed_expenses, precomputed_income, precomputed_base_expenses, precomputed_life_stages, precomputed_workation_costs = _precompute_monthly_cashflows(
        years_offset, remaining_months, config, post_fire_income,
        override_start_ages=override_start_ages,
        post_fire_income_override=post_fire_income_override,
    )

    # ステップ3.5: チャートと同一のコードパスでベースラインを取得
    # simulate_future_assets を force_fire_month 付きで実行し、
    # チャート表示のベースラインと完全に一致する資産パスを取得する
    print("  Step 3.5: Computing baseline asset path (chart-identical)...")
    baseline_df = simulate_future_assets(
        current_cash=current_cash,
        current_stocks=current_stocks,
        config=config,
        scenario=scenario,
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        override_start_ages=override_start_ages,
        force_fire_month=fire_month,
        extra_monthly_budget=extra_monthly_budget,
    )
    # FIRE月以降のデータを抽出（FIRE月 = ポストFIRE0ヶ月目）
    post_fire_rows = baseline_df.iloc[fire_month:fire_month + remaining_months]
    baseline_assets = post_fire_rows['assets'].tolist()
    # 早期終了（破綻）した場合は0で補完
    while len(baseline_assets) < remaining_months:
        baseline_assets.append(0)
    baseline_assets = baseline_assets[:remaining_months]
    # 線形償却フロアを適用: ベースラインが破綻した場合でもドローダウン検知を継続する
    baseline_assets = _apply_baseline_floor(baseline_assets, fire_cash + fire_stocks, remaining_months)

    print(f"  Baseline final assets: JPY{baseline_assets[-1]:,.0f}")
    print(f"  Baseline initial (FIRE): JPY{fire_cash + fire_stocks:,.0f}")
    print(f"  Baseline months simulated: {len(baseline_assets)}")
    print(f"  Baseline 10yr: JPY{baseline_assets[min(120, len(baseline_assets)-1)]:,.0f}")
    print(f"  Baseline first 5 months: {[f'{x/10000:.1f}' for x in baseline_assets[:5]]}")
    print(f"  Baseline last 5 months: {[f'{x/10000:.1f}' for x in baseline_assets[-5:]]}")

    # ステップ4: モンテカルロシミュレーション（FIRE後のみ）
    results = []
    all_timeseries = []  # 各イテレーションの月ごとデータ
    params = config['simulation'][scenario]
    mc_config = config['simulation']['monte_carlo']
    return_std_dev = mc_config['return_std_dev']
    mean_reversion_speed = mc_config['mean_reversion_speed']

    # 補填統計を記録（各月で補填が発生した回数）
    fill_count_by_month = np.zeros(remaining_months)
    fill_amount_by_month = np.zeros(remaining_months)

    for i in range(iterations):
        # 進捗表示
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{iterations} iterations completed")

        # FIRE後の期間分のランダムリターンを生成
        # 拡張モデル（GARCH + 非対称多期間平均回帰）が有効か確認
        enhanced_enabled = mc_config['enhanced_model']['enabled']

        if enhanced_enabled:
            # 拡張モデル: GARCH(1,1) + 非対称多期間平均回帰
            random_returns = generate_returns_enhanced(
                annual_return_mean=params['annual_return_rate'],
                annual_return_std=return_std_dev,
                total_months=remaining_months,
                config=config,
                random_seed=i
            )
        else:
            # 標準モデル: AR(1)平均回帰（後方互換性）
            random_returns = generate_random_returns(
                params['annual_return_rate'],
                return_std_dev,
                remaining_months,
                mean_reversion_speed=mean_reversion_speed,
                random_seed=i
            )

        timeseries = _simulate_post_fire_with_random_returns(
            current_cash=fire_cash,
            current_stocks=fire_stocks,
            years_offset=years_offset,
            config=config,
            scenario=scenario,
            random_returns=random_returns,
            nisa_balance=fire_nisa,
            nisa_cost_basis=fire_nisa_cost,
            stocks_cost_basis=fire_stocks_cost,
            return_timeseries=True,
            precomputed_expenses=precomputed_expenses,
            precomputed_income=precomputed_income,
            precomputed_base_expenses=precomputed_base_expenses,
            precomputed_life_stages=precomputed_life_stages,
            precomputed_workation_costs=precomputed_workation_costs,
            baseline_assets=baseline_assets,
            override_start_ages=override_start_ages,
            extra_monthly_budget=extra_monthly_budget,
        )

        final_assets = timeseries[-1] if len(timeseries) > 0 else 0
        all_timeseries.append(timeseries)

        # 成功判定: 資産がゼロ以上なら成功
        success = final_assets > 0

        results.append({
            'final_assets': final_assets,
            'success': success
        })

    # 統計情報を計算
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / iterations
    final_assets_list = [r['final_assets'] for r in results]

    # 月ごとの統計を計算（パーセンタイルを使用）
    # 対数正規分布は非対称なので、median±σ ではなくパーセンタイルを使う
    all_timeseries_array = np.array(all_timeseries)  # shape: (iterations, months)
    monthly_p50 = np.percentile(all_timeseries_array, 50, axis=0)   # 中央値
    monthly_p025 = np.percentile(all_timeseries_array, 2.5, axis=0) # 2σ下限（約95%信頼区間）
    monthly_p16 = np.percentile(all_timeseries_array, 16, axis=0)   # 1σ下限（約68%信頼区間）
    monthly_p84 = np.percentile(all_timeseries_array, 84, axis=0)   # 1σ上限
    monthly_p975 = np.percentile(all_timeseries_array, 97.5, axis=0) # 2σ上限

    # 追加リスクメトリクス
    safety_margin = config['post_fire_cash_strategy']['safety_margin']
    percentile_5 = float(np.percentile(final_assets_list, 5))
    bankruptcy_count = sum(1 for ts in all_timeseries if min(ts) <= safety_margin)
    bankruptcy_rate = bankruptcy_count / iterations

    monthly_p5 = np.percentile(all_timeseries_array, 5, axis=0)

    depletion_month_p5 = None
    for m_idx, val in enumerate(monthly_p5):
        if val <= safety_margin:
            depletion_month_p5 = m_idx
            break

    print(f"[OK] Monte Carlo simulation complete!")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Median final assets: JPY{np.median(final_assets_list):,.0f}")
    print(f"  P5 final assets: JPY{percentile_5:,.0f}")
    print(f"  2.5th percentile: JPY{np.percentile(final_assets_list, 2.5):,.0f}")
    print(f"  97.5th percentile: JPY{np.percentile(final_assets_list, 97.5):,.0f}")
    print(f"  Mean final assets: JPY{np.mean(final_assets_list):,.0f}")
    print(f"  Bankruptcy rate: {bankruptcy_rate*100:.1f}%")

    return {
        'success_rate': success_rate,
        'median_final_assets': np.median(final_assets_list),
        'mean_final_assets': np.mean(final_assets_list),
        'percentile_5': percentile_5,
        'percentile_2_5': float(np.percentile(final_assets_list, 2.5)),
        'percentile_97_5': float(np.percentile(final_assets_list, 97.5)),
        'bankruptcy_rate': bankruptcy_rate,
        'depletion_month_p5': depletion_month_p5,
        'all_results': results,
        'fire_month': fire_month,
        'monthly_p50': monthly_p50,
        'monthly_p5': monthly_p5,
        'monthly_p025': monthly_p025,
        'monthly_p16': monthly_p16,
        'monthly_p84': monthly_p84,
        'monthly_p975': monthly_p975,
    }


def _sell_stocks_vectorized(
    shortage: np.ndarray,
    stocks: np.ndarray,
    nisa_balance: np.ndarray,
    nisa_cost_basis: np.ndarray,
    stocks_cost_basis: np.ndarray,
    capital_gains_tax_rate: float,
    allocation_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    _sell_stocks_with_tax の NumPy ベクトル化版。N 個の MC iteration を同時処理する。

    Args:
        shortage: 各 iteration で確保したい金額 [N]
        stocks: 株式残高（NISA含む）[N]
        nisa_balance: NISA 残高 [N]
        nisa_cost_basis: NISA 簿価 [N]
        stocks_cost_basis: 株式全体の簿価 [N]
        capital_gains_tax_rate: 譲渡益税率（スカラー）
        allocation_enabled: 資産配分有効フラグ（スカラー）

    Returns:
        (nisa_sold, cash_from_taxable, capital_gain,
         new_stocks, new_nisa_balance, new_nisa_cost_basis, new_stocks_cost_basis)
        すべて shape [N]

    呼び出し元での現金加算:
        最低現金型 (_manage_post_fire_cash): cash += nisa_sold + cash_from_taxable
        支出不足型 (expense payment):        cash += cash_from_taxable のみ
    """
    N = len(shortage)

    if not allocation_enabled:
        sold = np.minimum(shortage, stocks)
        return (
            np.zeros(N),
            sold,
            np.zeros(N),
            stocks - sold,
            nisa_balance.copy(),
            nisa_cost_basis.copy(),
            np.maximum(0.0, stocks_cost_basis - sold),
        )

    # ── NISA 優先売却（非課税）
    nisa_to_sell = np.minimum(shortage, np.maximum(0.0, nisa_balance))
    safe_nisa = np.maximum(nisa_balance, 1e-10)
    nisa_sold_cost = np.where(nisa_balance > 0,
                               nisa_to_sell / safe_nisa * nisa_cost_basis,
                               0.0)
    remaining = shortage - nisa_to_sell

    new_nisa_balance = np.maximum(0.0, nisa_balance - nisa_to_sell)
    new_nisa_cost_basis = np.maximum(0.0, nisa_cost_basis - nisa_sold_cost)
    stocks_after_nisa = stocks - nisa_to_sell
    costs_after_nisa = np.maximum(0.0, stocks_cost_basis - nisa_sold_cost)

    # ── 課税口座売却（税引後で remaining を確保）
    taxable_stocks = np.maximum(0.0, stocks_after_nisa - new_nisa_balance)
    has_taxable = (remaining > 0) & (taxable_stocks > 0)
    safe_taxable = np.maximum(taxable_stocks, 1e-10)

    avg_cost = np.where(
        has_taxable,
        (costs_after_nisa - new_nisa_cost_basis) / safe_taxable,
        0.0,
    )
    gain_ratio = np.maximum(0.0, 1.0 - avg_cost)
    eff_tax = capital_gains_tax_rate * gain_ratio
    req_sale = np.where(
        eff_tax < 1.0,
        remaining / np.maximum(1.0 - eff_tax, 1e-10),
        remaining,
    )
    taxable_sold = np.where(has_taxable, np.minimum(req_sale, taxable_stocks), 0.0)

    sale_cost = taxable_sold * avg_cost
    capital_gain = np.maximum(0.0, taxable_sold - sale_cost)
    tax = capital_gain * capital_gains_tax_rate
    cash_from_taxable = taxable_sold - tax

    new_stocks = stocks_after_nisa - taxable_sold
    new_stocks_cost = np.maximum(0.0, costs_after_nisa - sale_cost)

    return (
        nisa_to_sell,
        cash_from_taxable,
        capital_gain,
        new_stocks,
        new_nisa_balance,
        new_nisa_cost_basis,
        new_stocks_cost,
    )
