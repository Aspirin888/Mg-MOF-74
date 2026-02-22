import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sko.GA import GA
import joblib
from catboost import CatBoostRegressor  # 导入 CatBoost
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="Mg-MOF-74 吸附剂逆向设计", layout="wide")
st.title("Mg-MOF-74 吸附剂逆向设计平台")
st.markdown("基于已训练的 CatBoost 模型，通过遗传算法逆向搜索满足目标吸附容量和环境条件的材料参数。")

# ========== 加载模型和特征信息 ==========
@st.cache_resource
def load_model_and_info():
    # 加载 CatBoost 模型（假设用 joblib 保存）
    # 注意：CatBoost 原生保存格式可能不同，这里按 joblib 处理
    model = joblib.load('CatBoost.pkl')
    scaler = joblib.load('scaler.pkl')
    info = joblib.load('model_info.pkl')
    return model, scaler, info

model, scaler, info = load_model_and_info()

# 从 info 中提取变量
numeric_cols = info['numeric_cols']
fixed_cols = info['fixed_cols']
cat_vars = info['cat_vars']          # 字典：分类变量名 -> 类别列表
all_cols = info['all_cols']
numeric_bounds = info['numeric_bounds']
ratio_bounds = info['ratio_bounds']  # (低, 高)

# ========== 定义约束和目标函数 ==========
def constraint_penalty(x):
    """SBET/Vpore 比值约束的惩罚项"""
    sbet = x[1]   # SBET_m2_g 是第二个连续变量（索引1）
    vpore = x[2]  # Vpore_cm3_g 是第三个
    ratio = sbet / (vpore + 1e-6)
    q_low, q_high = ratio_bounds
    penalty = 0.0
    if ratio < q_low:
        penalty = 1000 * (q_low - ratio)
    elif ratio > q_high:
        penalty = 1000 * (ratio - q_high)
    return penalty

def objective_func(x, T_target, P_target, Q_target, mode):
    """遗传算法目标函数"""
    n_cont = len(numeric_cols)
    cont_vals = x[:n_cont]
    cat_indices = x[n_cont:].astype(int)

    # 索引合法性修正
    for j, (var, cats) in enumerate(cat_vars.items()):
        if cat_indices[j] < 0 or cat_indices[j] >= len(cats):
            cat_indices[j] = np.clip(cat_indices[j], 0, len(cats)-1)

    # 构建特征向量
    x_vec = np.zeros(len(all_cols))

    # 填充连续变量
    for i, col in enumerate(numeric_cols):
        idx = all_cols.index(col)
        x_vec[idx] = cont_vals[i]

    # 填充固定温度压力
    idx_p = all_cols.index('Pressure_MPa')
    idx_t = all_cols.index('Temperature_K')
    x_vec[idx_p] = P_target
    x_vec[idx_t] = T_target

    # 填充分类变量（独热编码）
    for j, (var, cats) in enumerate(cat_vars.items()):
        cat_name = cats[cat_indices[j]]
        onehot_col = f"{var}_{cat_name}"
        if onehot_col in all_cols:
            idx = all_cols.index(onehot_col)
            x_vec[idx] = 1.0

    # 标准化并预测
    x_scaled = scaler.transform(x_vec.reshape(1, -1))
    pred = model.predict(x_scaled)[0]

    penalty = constraint_penalty(cont_vals)

    if mode == 'target':
        return abs(pred - Q_target) + penalty
    else:  # mode == 'max'
        return -pred + penalty

# ========== 侧边栏输入 ==========
st.sidebar.header("环境条件设置")
T_target = st.sidebar.number_input("温度 (K)", value=298.0, step=1.0)
P_target = st.sidebar.number_input("压力 (MPa)", value=1.0, step=0.1)

mode = st.sidebar.radio(
    "优化模式",
    ("target", "max"),
    format_func=lambda x: "目标吸附容量" if x == "target" else "最大化吸附容量"
)
if mode == "target":
    Q_target = st.sidebar.number_input("目标吸附容量 (mmol/g)", value=3.5, step=0.1)
else:
    Q_target = None

top_n = st.sidebar.slider("输出候选数量", min_value=1, max_value=10, value=5)
run_opt = st.sidebar.button("开始优化")

# ========== 运行优化 ==========
if run_opt:
    with st.spinner("正在运行遗传算法优化，请稍候..."):
        n_cont = len(numeric_cols)
        n_cat = len(cat_vars)
        n_dim = n_cont + n_cat

        # 变量边界
        lb = [numeric_bounds[col][0] for col in numeric_cols] + [0] * n_cat
        ub = [numeric_bounds[col][1] for col in numeric_cols] + [len(cat_vars[var]) - 0.5 for var in cat_vars]

        # 创建遗传算法对象
        ga = GA(
            func=lambda x: objective_func(x, T_target, P_target, Q_target, mode),
            n_dim=n_dim,
            size_pop=100,
            max_iter=200,
            lb=lb,
            ub=ub,
            precision=1e-7
        )

        best_x, best_y = ga.run()

        # 获取多个候选解
        final_X = ga.X
        final_Y = ga.Y
        sorted_idx = np.argsort(final_Y)
        top_idx = sorted_idx[:top_n]
        top_x = final_X[top_idx]
        top_y = final_Y[top_idx]

        # 解析候选解
        candidates = []
        for k, (x, y) in enumerate(zip(top_x, top_y)):
            cont_vals = x[:n_cont]
            cat_indices = x[n_cont:].astype(int)
            for j, (var, cats) in enumerate(cat_vars.items()):
                if cat_indices[j] >= len(cats):
                    cat_indices[j] = len(cats) - 1

            result = {}
            for i, col in enumerate(numeric_cols):
                result[col] = cont_vals[i]
            for j, (var, cats) in enumerate(cat_vars.items()):
                result[var] = cats[cat_indices[j]]

            # 验证预测值（不含惩罚）
            x_vec = np.zeros(len(all_cols))
            for i, col in enumerate(numeric_cols):
                idx = all_cols.index(col)
                x_vec[idx] = cont_vals[i]
            idx_p = all_cols.index('Pressure_MPa')
            idx_t = all_cols.index('Temperature_K')
            x_vec[idx_p] = P_target
            x_vec[idx_t] = T_target
            for j, (var, cats) in enumerate(cat_vars.items()):
                cat_name = cats[cat_indices[j]]
                onehot_col = f"{var}_{cat_name}"
                if onehot_col in all_cols:
                    idx = all_cols.index(onehot_col)
                    x_vec[idx] = 1.0
            x_scaled = scaler.transform(x_vec.reshape(1, -1))
            pred = model.predict(x_scaled)[0]

            candidates.append((result, pred, y))

        # 保存到 session state 以便展示
        st.session_state['candidates'] = candidates
        st.session_state['mode'] = mode
        st.session_state['Q_target'] = Q_target if mode == 'target' else None

# ========== 显示结果 ==========
if 'candidates' in st.session_state:
    candidates = st.session_state['candidates']
    mode = st.session_state['mode']
    Q_target = st.session_state['Q_target']

    st.header("优化结果")
    st.subheader(f"前 {len(candidates)} 组最优候选解")

    # 转换为 DataFrame
    df_candidates = pd.DataFrame([r[0] for r in candidates])
    df_candidates['Predicted_Adsorption'] = [r[1] for r in candidates]
    df_candidates['Objective'] = [r[2] for r in candidates]

    # 显示表格
    st.dataframe(
        df_candidates.style.format(
            {col: "{:.4f}" for col in numeric_cols + ['Predicted_Adsorption', 'Objective']}
        )
    )

    # 下载按钮
    csv = df_candidates.to_csv(index=False).encode('utf-8')
    st.download_button("下载候选结果为CSV", data=csv, file_name="candidates.csv", mime="text/csv")

    # ========== 可视化 ==========
    st.subheader("可视化分析")

    # 设置学术绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 150

    # 1. 平行坐标图
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    parallel_vars = numeric_cols + ['Predicted_Adsorption']
    pd.plotting.parallel_coordinates(
        df_candidates[parallel_vars],
        'Predicted_Adsorption',
        colormap='viridis',
        linewidth=2,
        alpha=0.8
    )
    ax1.set_title('Parallel Coordinates of Top Candidates')
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Normalized Value')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Predicted (mmol/g)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig1)

    # 2. 散点图矩阵
    fig2 = plt.figure(figsize=(10, 8))
    g = sns.pairplot(
        df_candidates,
        vars=numeric_cols + ['Predicted_Adsorption'],
        diag_kind='hist',
        plot_kws={'alpha': 0.8, 's': 60},
        diag_kws={'alpha': 0.7, 'bins': 10}
    )
    g.fig.suptitle('Scatter Matrix of Continuous Variables', y=1.02, fontsize=16)
    st.pyplot(g.fig)

    # 3. 分类变量分布
    cat_vars_names = list(cat_vars.keys())
    fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, var in enumerate(cat_vars_names):
        ax = axes[idx]
        counts = df_candidates[var].value_counts()
        colors = sns.color_palette("viridis", len(counts))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
        ax.set_title(f'{var} Distribution')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha='center',
                va='bottom',
                fontsize=10
            )
    # 隐藏多余的子图（如果分类变量少于4）
    for j in range(len(cat_vars_names), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig3)

    # 4. 预测值与目标对比
    if mode == 'target':
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.scatter(
            [Q_target] * len(df_candidates),
            df_candidates['Predicted_Adsorption'],
            s=120,
            c='red',
            alpha=0.7,
            edgecolor='black',
            zorder=5
        )
        ax4.plot(
            [Q_target - 0.2, Q_target + 0.2],
            [Q_target - 0.2, Q_target + 0.2],
            'k--',
            linewidth=2,
            label='Perfect match'
        )
        ax4.set_xlabel('Target Adsorption (mmol/g)')
        ax4.set_ylabel('Predicted Adsorption (mmol/g)')
        ax4.set_title('Predicted vs Target')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.bar(
            range(1, len(df_candidates) + 1),
            df_candidates['Predicted_Adsorption'],
            color='steelblue',
            edgecolor='black'
        )
        ax4.set_xlabel('Candidate Rank')
        ax4.set_ylabel('Predicted Adsorption (mmol/g)')
        ax4.set_title('Top Candidates - Predicted Adsorption')
        ax4.set_xticks(range(1, len(df_candidates) + 1))
        ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig4)

else:
    st.info("请在侧边栏设置参数并点击「开始优化」按钮。")
