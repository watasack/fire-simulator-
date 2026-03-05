"""
FIREシミュレーター 無料デモ版

- 年収・支出・現在資産の3入力のみ
- FIRE到達年齢のみ表示
- 詳細グラフ・FIRE成功確率はフル版（購入）で利用可能
"""

import sys
import copy
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.simulator import simulate_future_assets

# -------------------------------------------------------------------
# ページ設定
# -------------------------------------------------------------------
st.set_page_config(
    page_title="共働き・子育て世帯のFIREシミュレーター（無料デモ）",
    page_icon="🔥",
    layout="centered",
)

# -------------------------------------------------------------------
# ヘッダー
# -------------------------------------------------------------------
st.title("🔥 共働き・子育て世帯のFIREシミュレーター")
st.caption("入力データはサーバーに保存されません。ブラウザ内のみで計算します。")

st.markdown("""
**共働き・子育て世帯**向けのFIREシミュレーターです。
夫婦それぞれの収入・支出・資産を入力するだけで、推計FIRE到達年齢がわかります。
""")

st.divider()

# -------------------------------------------------------------------
# 入力フォーム
# -------------------------------------------------------------------

st.subheader("💰 収入・支出・資産")

col1, col2 = st.columns(2)

with col1:
    income_husband = st.number_input(
        "夫の月手取り収入（万円）",
        min_value=0,
        max_value=200,
        value=45,
        step=1,
        help="税・社保引き後の手取り額。",
    )
    age_husband = st.number_input(
        "夫の現在年齢（歳）",
        min_value=20,
        max_value=65,
        value=35,
        step=1,
    )

with col2:
    income_wife = st.number_input(
        "妻の月手取り収入（万円）",
        min_value=0,
        max_value=200,
        value=35,
        step=1,
        help="税・社保引き後の手取り額。",
    )
    age_wife = st.number_input(
        "妻の現在年齢（歳）",
        min_value=20,
        max_value=65,
        value=33,
        step=1,
    )

total_income = income_husband + income_wife
current_age = age_husband
st.caption(f"世帯月手取り合計: **{total_income} 万円**（年収 {total_income * 12} 万円）")

col3, col4 = st.columns(2)
with col3:
    monthly_expense_man = st.number_input(
        "月間支出（万円）",
        min_value=5,
        max_value=150,
        value=28,
        step=1,
        help="生活費・住宅ローン・保険・教育費等の合計。",
    )
with col4:
    current_assets_man = st.number_input(
        "現在の金融資産（万円）",
        min_value=0,
        max_value=50000,
        value=500,
        step=50,
        help="現金・預金・投資信託・株式の合計。不動産は除く。",
    )

st.divider()

# -------------------------------------------------------------------
# 育休・時短設定（フル版ティザー）
# -------------------------------------------------------------------

with st.expander("👶 育休・時短設定（フル版で設定可能）"):
    st.info("育休・産休・時短勤務の期間と収入変化はフル版で設定できます。このデモでは収入一定として計算します。")
    col_a, col_b = st.columns(2)
    with col_a:
        st.number_input("妻の育休期間（ヶ月）", value=12, disabled=True)
        st.number_input("育休中の妻の月収（万円）", value=15, disabled=True)
    with col_b:
        st.number_input("時短勤務期間（ヶ月）", value=36, disabled=True)
        st.number_input("時短中の妻の月収（万円）", value=28, disabled=True)
    st.caption("🔒 育休・時短を反映したFIRE計算はフル版（¥3,000）で")

st.divider()

# -------------------------------------------------------------------
# 計算ボタン
# -------------------------------------------------------------------
calc_button = st.button("FIRE到達年齢を計算する", type="primary", use_container_width=False)

if calc_button:
    # 単位変換（万円 → 円）
    monthly_income = total_income * 10000
    monthly_expense = monthly_expense_man * 10000
    annual_expense = monthly_expense * 12
    current_assets = current_assets_man * 10000

    # 資産を現金30% / 株式70% に分割
    current_cash = current_assets * 0.30
    current_stocks = current_assets * 0.70

    # -------------------------------------------------------------------
    # config.yaml を読み込み、デモ用にオーバーライド
    # -------------------------------------------------------------------
    try:
        with open(project_root / "demo_config.yaml", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("設定ファイルが見つかりません。開発者にお問い合わせください。")
        st.stop()

    cfg = copy.deepcopy(base_config)

    # --- 収入: 修平・桜の個別収入を0にして monthly_income パラメータに委譲 ---
    # _calculate_monthly_income の分岐: shuhei + sakura == 0 のとき monthly_income を使用
    cfg["simulation"]["shuhei_income"] = 0
    cfg["simulation"]["sakura_income"] = 0

    # --- 支出: カテゴリ別予算を無効化し、manual_annual_expense で固定 ---
    cfg["fire"]["expense_categories"]["enabled"] = False
    cfg["fire"]["manual_annual_expense"] = annual_expense

    # --- 年齢設定 ---
    cfg["simulation"]["start_age"] = int(current_age)

    # --- FIRE最適化月をリセット（自然なFIRE検出） ---
    cfg["fire"]["optimal_fire_month"] = None
    cfg["fire"]["optimal_extra_monthly_budget"] = 0

    # --- 産休・育休・時短を無効化（個人設定を排除） ---
    cfg["simulation"]["maternity_leave"] = []
    cfg["simulation"]["shuhei_parental_leave"] = []
    cfg["simulation"]["shuhei_reduced_hours"] = []

    # --- FIRE後支出のターゲット計算用（4%ルールの分母として使用される）を更新 ---
    # base_expense_by_stage['empty_nest'] は FIRE判定の目標資産算出に使われる
    for stage in cfg["fire"]["base_expense_by_stage"]:
        cfg["fire"]["base_expense_by_stage"][stage] = annual_expense

    # -------------------------------------------------------------------
    # シミュレーション実行
    # -------------------------------------------------------------------
    with st.spinner("計算中..."):
        try:
            df = simulate_future_assets(
                current_cash=current_cash,
                current_stocks=current_stocks,
                monthly_income=monthly_income,
                monthly_expense=monthly_expense,
                config=cfg,
                scenario="standard",
            )
        except Exception as e:
            st.error(f"シミュレーション中にエラーが発生しました: {e}")
            st.stop()

    # -------------------------------------------------------------------
    # 結果表示
    # -------------------------------------------------------------------
    fire_rows = df[df["fire_achieved"] == True]

    if len(fire_rows) > 0:
        fire_month = int(fire_rows.iloc[0]["fire_month"])
        fire_age = current_age + fire_month / 12
        years_to_fire = fire_month / 12

        st.success("✅ シミュレーション完了")

        # メトリクス表示
        m1, m2, m3 = st.columns(3)
        m1.metric("推計FIRE到達年齢", f"{fire_age:.0f} 歳")
        m2.metric("今から", f"{years_to_fire:.1f} 年後")
        fire_assets_man = fire_rows.iloc[0]["assets"] / 10000
        m3.metric("FIRE時の推計資産", f"{fire_assets_man:.0f} 万円")

        st.divider()

        # ぼかしグラフ（デモ制限）
        st.subheader("📊 資産推移（一部プレビュー）")

        preview_months = min(60, len(df))  # 最初の5年分だけ表示
        df_preview = df.iloc[:preview_months]

        fig = go.Figure()

        # プレビュー部分（薄く表示）
        fig.add_trace(go.Scatter(
            x=df_preview["date"].astype(str),
            y=df_preview["assets"] / 10000,
            mode="lines",
            line=dict(color="rgba(100, 149, 237, 0.4)", width=2),
            name="資産推移（5年プレビュー）",
        ))

        # ロック表示
        fig.add_annotation(
            text="🔒 全期間グラフ・FIRE成功確率・最悪ケース分析はフル版で",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        )

        fig.update_layout(
            xaxis_title="年月",
            yaxis_title="資産（万円）",
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch")

        # フル版の差別化ポイント
        st.info("""
**フル版でできること（このデモではできないこと）：**

- 📈 FIRE成功確率（1,000通りの未来を計算）
- 📉 最悪ケース：リターンが悪い年が続いても大丈夫か？
- 👶 育休・産休・時短勤務の正確な反映
- 💑 夫婦別収入・FIRE後副収入の設定
- 📊 全期間の資産推移グラフ
- 🏦 NISA・iDeCo・年金の最適化
        """)

    else:
        st.warning("""
⚠️ シミュレーション期間内にFIRE到達できませんでした。

以下を確認してください：
- 月間支出を下げる（支出の削減がFIREへの最短ルート）
- 年収を増やす想定にする
- 現在資産を増やす

フル版では最悪ケース・感度分析で「何を変えればFIREできるか」を詳細分析できます。
        """)

    st.divider()

    # 購入CTA（常に表示）
    st.subheader("フル版を購入する")
    st.markdown("""
育休・産休の影響、FIRE成功確率、最悪ケース分析を含む完全版。
**買い切り ¥3,000 / Googleスプレッドシート版付き**
    """)

    col_buy, col_spacer = st.columns([2, 1])
    with col_buy:
        # Gumroad URL は後で差し替え
        st.link_button(
            "🛒 フル版を購入する（¥3,000）",
            url="https://firewithkids.gumroad.com/l/okjjyz",
            use_container_width=False,
        )

    st.divider()

    # メール登録（補助導線）
    st.subheader("📧 育休・共働きシナリオの使い方ガイドを無料で受け取る")
    st.caption("育休12ヶ月・時短勤務など共働き世帯向けの設定例をメールでお送りします。購入前の参考にどうぞ。")
    # TODO: Googleフォーム作成後にURLを差し替え
    st.link_button(
        "無料ガイドを受け取る（メール登録）",
        url="https://forms.gle/mqrmGXNzy73Q1qb67",
        use_container_width=False,
    )

# -------------------------------------------------------------------
# フッター
# -------------------------------------------------------------------
st.divider()
st.caption("""
⚠️ 本シミュレーターは情報提供のみを目的としており、投資アドバイスではありません。
数理モデリングの専門家が自家用に開発したツールを汎用化したものです。
投資判断はご自身の責任で行ってください。
""")
