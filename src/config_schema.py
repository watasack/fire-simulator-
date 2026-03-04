"""
config.yaml の Pydantic スキーマ定義

全フィールドを必須とし、config.yaml に未記載のパラメータがあれば
load_config() 時に明確なエラーを返す。
Optional なフィールドは正当な理由があるもののみ（最適化結果、明示的な null）。
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator


class IncomeForecastConfig(BaseModel):
    include_keywords: List[str]
    exclude_keywords: List[str]


class DataConfig(BaseModel):
    encoding: str
    asset_file: str
    transaction_pattern: str
    income_forecast: IncomeForecastConfig


class MaternityLeaveEntry(BaseModel):
    child: str
    months_before: int
    months_after: int
    monthly_income: int


class ParentalLeaveEntry(BaseModel):
    child: str
    months_after: int
    monthly_income: int
    monthly_income_after_180days: int


class ReducedHoursEntry(BaseModel):
    child: str
    start_months_after: int
    end_months_after: int
    income_ratio: float


class StandardScenarioConfig(BaseModel):
    annual_return_rate: float
    inflation_rate: float
    income_growth_rate: float
    expense_growth_rate: float
    pension_growth_rate: float


class EnhancedModelConfig(BaseModel):
    enabled: bool
    garch_omega: float
    garch_alpha: float
    garch_beta: float
    volatility_floor: float
    volatility_ceiling: float
    mean_reversion_window: int
    mr_speed_crash: float
    mr_speed_normal: float
    mr_speed_bubble: float
    crash_threshold: float
    bubble_threshold: float


class MonteCarloConfig(BaseModel):
    enabled: bool
    iterations: int
    return_std_dev: float
    mean_reversion_speed: float
    enhanced_model: EnhancedModelConfig


class SimulationConfig(BaseModel):
    years: int
    start_age: int
    life_expectancy: int
    shuhei_income: int
    sakura_income: int
    initial_labor_income: int
    shuhei_post_fire_income: int
    sakura_post_fire_income: int
    maternity_leave: List[MaternityLeaveEntry]
    shuhei_parental_leave: List[ParentalLeaveEntry]
    shuhei_reduced_hours: List[ReducedHoursEntry]
    standard: StandardScenarioConfig
    monte_carlo: MonteCarloConfig


class ExpenseCategoryDefinition(BaseModel):
    id: str
    name: str
    discretionary: bool
    description: str


class StageBudget(BaseModel):
    food_home: int
    utilities_electricity: int
    utilities_gas: int
    utilities_water: int
    communication_mobile: int
    communication_internet: int
    transport_commute: int
    insurance_life: int
    insurance_casualty: int
    medical: int
    household_goods: int
    food_out: int
    transport_other: int
    clothing: int
    entertainment_movies: int
    entertainment_other: int
    travel: int
    hobby: int
    education_books: int
    education_courses: int
    beauty: int
    other: int


class ExpenseCategoriesConfig(BaseModel):
    enabled: bool
    definitions: List[ExpenseCategoryDefinition]
    budgets_by_stage: Dict[str, StageBudget]


class BaseExpenseByStage(BaseModel):
    young_child: int
    elementary: int
    junior_high: int
    high_school: int
    university: int
    empty_nest: int
    empty_nest_active: int
    empty_nest_senior: int
    empty_nest_elderly: int


class DiscretionaryRatioByStage(BaseModel):
    young_child: float
    elementary: float
    junior_high: float
    high_school: float
    university: float
    empty_nest: float
    empty_nest_active: float
    empty_nest_senior: float
    empty_nest_elderly: float


class EmptyNestSubStages(BaseModel):
    senior_from_age: int
    elderly_from_age: int


class DynamicExpenseReductionConfig(BaseModel):
    enabled: bool
    surplus_spending_rate: float
    max_cut_ratio: float
    max_boost_ratio: float


class AdditionalChildExpenseByStage(BaseModel):
    young_child: int
    elementary: int
    junior_high: int
    high_school: int
    university: int
    empty_nest: int


class FireConfig(BaseModel):
    expense_categories: ExpenseCategoriesConfig
    base_expense_by_stage: BaseExpenseByStage
    discretionary_ratio_by_stage: DiscretionaryRatioByStage
    empty_nest_sub_stages: EmptyNestSubStages
    dynamic_expense_reduction: DynamicExpenseReductionConfig
    additional_child_expense_by_stage: AdditionalChildExpenseByStage
    optimal_fire_month: Optional[int] = None
    optimal_extra_monthly_budget: Optional[int] = None
    manual_annual_expense: Optional[int] = None


class ChildConfig(BaseModel):
    name: str
    birthdate: str
    nursery: str
    kindergarten: str
    elementary: str
    junior_high: str
    high: str
    university: str


class NurseryCosts(BaseModel):
    standard: int


class TwoTierCosts(BaseModel):
    public: int
    private: int


class UniversityCosts(BaseModel):
    national: int
    private_arts: int
    private_science: int


class EducationCosts(BaseModel):
    nursery: NurseryCosts
    kindergarten: TwoTierCosts
    elementary: TwoTierCosts
    junior_high: TwoTierCosts
    high: TwoTierCosts
    university: UniversityCosts


class EducationConfig(BaseModel):
    enabled: bool
    children: List[ChildConfig]
    costs: EducationCosts


class ChildAllowanceConfig(BaseModel):
    enabled: bool
    first_child_under_3: int
    second_child_plus_under_3: int
    age_3_to_high_school: int


class MortgageConfig(BaseModel):
    enabled: bool
    monthly_payment: int
    end_date: str


class MaintenanceItem(BaseModel):
    name: str
    cost: int
    frequency_years: int
    first_year: int


class HouseMaintenanceConfig(BaseModel):
    enabled: bool
    items: List[MaintenanceItem]


class PensionPerson(BaseModel):
    name: str
    birthdate: str
    pension_type: Literal['employee', 'national']
    work_start_age: Optional[int] = None
    avg_monthly_salary: Optional[int] = None
    past_pension_base_annual: Optional[int] = None
    past_contribution_months: Optional[int] = None
    override_start_age: Optional[int] = None
    annual_amount: Optional[int] = None

    @model_validator(mode='after')
    def validate_employee_fields(self):
        if self.pension_type == 'employee':
            required = {
                'work_start_age': self.work_start_age,
                'avg_monthly_salary': self.avg_monthly_salary,
                'past_pension_base_annual': self.past_pension_base_annual,
                'past_contribution_months': self.past_contribution_months,
            }
            missing = [k for k, v in required.items() if v is None]
            if missing:
                raise ValueError(
                    f"pension_type='employee' requires: {', '.join(missing)}"
                )
        return self


class PensionConfig(BaseModel):
    enabled: bool
    start_age: int
    people: List[PensionPerson]


class SocialInsuranceConfig(BaseModel):
    enabled: bool
    national_pension_monthly_premium: int
    health_insurance_income_rate: float
    health_insurance_per_person: int
    health_insurance_per_household: int
    health_insurance_members: int
    health_insurance_basic_deduction: int
    health_insurance_max_premium: int


class WorkationConfig(BaseModel):
    enabled: bool
    start_child_index: int
    start_child_age: int
    annual_cost: int


class AssetAllocationConfig(BaseModel):
    enabled: bool
    cash_buffer_months: int
    auto_invest_threshold: float
    nisa_enabled: bool
    nisa_annual_limit: int
    invest_beyond_nisa: bool
    min_cash_balance: int
    capital_gains_tax_rate: float


class PostFireCashStrategyConfig(BaseModel):
    enabled: bool
    safety_margin: int
    target_cash_reserve: int
    monthly_buffer_months: int
    market_crash_threshold: float
    recovery_threshold: float
    emergency_cash_floor: int


class PensionDeferralConfig(BaseModel):
    enabled: bool
    defer_to_70_threshold: float
    defer_to_68_threshold: float
    early_at_62_threshold: float
    deferral_increase_rate: float
    early_decrease_rate: float
    min_start_age: int
    max_start_age: int


class OptimizationConfig(BaseModel):
    min_asset_floor: int
    austerity_years_before_pension: int
    pre_pension_reduction_candidates: List[float]
    pre_pension_reduction_rate: Optional[float] = None


class VisualizationConfig(BaseModel):
    font_family: str


class OutputConfig(BaseModel):
    html_file: str


class AppConfig(BaseModel):
    data: DataConfig
    simulation: SimulationConfig
    fire: FireConfig
    education: EducationConfig
    child_allowance: ChildAllowanceConfig
    mortgage: MortgageConfig
    house_maintenance: HouseMaintenanceConfig
    pension: PensionConfig
    social_insurance: SocialInsuranceConfig
    workation: WorkationConfig
    asset_allocation: AssetAllocationConfig
    post_fire_cash_strategy: PostFireCashStrategyConfig
    pension_deferral: PensionDeferralConfig
    optimization: OptimizationConfig
    visualization: VisualizationConfig
    output: OutputConfig
