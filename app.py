import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sko.GA import GA
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置页面标题和布局
st.set_page_config(page_title="Mg-MOF-74 吸附剂逆向设计", layout="wide")
st.title("Mg-MOF-74 吸附剂逆向设计平台")
st.markdown("基于已训练的 Extra Trees 模型，通过遗传算法逆向搜索满足目标吸附容量和环境条件的材料参数。")

# ========== 缓存数据加载和特征提取 ==========
@st.cache_data
def load_data_and_features():
    df = pd.read_csv('Mg_MOF_74_独热编码.csv')
    X = df.drop('Adsorption_capacity_mmol_g', axis=1)
    y = df['Adsorption_capacity_mmol_g']
    
    # 划分训练集和测试集（与模型训练一致）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 特征列
    numeric_cols = ['Molar ratio', 'SBET_m2_g', 'Vpore_cm3_g']
    fixed_cols = ['Pressure_MPa', 'Temperature_K']
    cat_vars_names = ['Mg_source', 'Solvent', 'Treatment', 'Morphology']
    
    all_cols = X.columns.tolist()
    
    # 解析分类变量类别
    cat_vars = {name: [] for name in cat_vars_names}
    for col in all_cols:
        for var in cat_vars_names:
            if col.startswith(var + '_'):
                cat_name = col[len(var)+1:]
                if cat_name not in cat_vars[var]:
                    cat_vars[var].append(cat_name)
                break
    for var in cat_vars:
        cat_vars[var].sort()
    
    # 单变量边界（基于训练数据）
    numeric_bounds = {}
    for col in numeric_cols:
        numeric_bounds[col] = (X_train[col].min(), X_train[col].max())
    
    # SBET/Vpore 比值约束
    ratio_train = X_train['SBET_m2_g'] / (X_train['Vpore_cm3_g'] + 1e-6)
    q_low = ratio_train.quantile(0.025)
    q_high = ratio_train.quantile(0.975)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'fixed_cols': fixed_cols,
        'cat_vars': cat_vars,
        'all_cols': all_cols,
        'numeric_bounds': numeric_bounds,
        'ratio_bounds': (q_low, q_high),
    }

data_info = load_data_and_features()

# ========== 加载模型 ==========
@st.cache_resource
def load_model():
    model = joblib.load('XGBoost.pkl')
    return model

model = load_model()

# ========== 定义目标函数和约束 ==========
def constraint_penalty(x, data_info):
    sbet = x[1]  # SBET_m2_g 是第二个连续变量
    vpore = x[2]  # Vpore_cm3_g 是第三个
    ratio = sbet / (vpore + 1e-6)
    q_low, q_high = data_info['ratio_bounds']
    penalty = 0.0
    if ratio < q_low:
        penalty = 1000 * (q_low - ratio)
    elif ratio > q_high:
        penalty = 1000 * (ratio - q_high)
    return penalty

def objective_func(x, T_target, P_target, Q_target, mode, data_info, model, scaler):
    n_cont = len(data_info['numeric_cols'])
    cont_vals = x[:n_cont]
    cat_indices = x[n_cont:].astype(int)

    # 索引修正
    cat_vars = data_info['cat_vars']
    for j, (var, cats) in enumerate(cat_vars.items()):
        if cat_indices[j] < 0 or cat_indices[j] >= len(cats):
            cat_indices[j] = np.clip(cat_indices[j], 0, len(cats)-1)

    x_vec = np.zeros(len(data_info['all_cols']))
    feature_columns = data_info['all_cols']

    # 填充连续变量
    for i, col in enumerate(data_info['numeric_cols']):
        idx = feature_columns.index(col)
        x_vec[idx] = cont_vals[i]

    # 固定温度压力
    idx_p = feature_columns.index('Pressure_MPa')
    idx_t = feature_columns.index('Temperature_K')
    x_vec[idx_p] = P_target
    x_vec[idx_t] = T_target

    # 填充分类变量
    for j, (var, cats) in enumerate(cat_vars.items()):
        cat_name = cats[cat_indices[j]]
        onehot_col = f"{var}_{cat_name}"
        if onehot_col in feature_columns:
            idx = feature_columns.index(onehot_col)
            x_vec[idx] = 1.0

    x_scaled = scaler.transform(x_vec.reshape(1, -1))
    pred = model.predict(x_scaled)[0]

    penalty = constraint_penalty(cont_vals, data_info)

    if mode == 'target':
        return abs(pred - Q_target) + penalty
    else:  # mode == 'max'
        return -pred + penalty

# ========== 侧边栏输入 ==========
st.sidebar.header("环境条件设置")
T_target = st.sidebar.number_input("温度 (K)", value=298.0, step=1.0)
P_target = st.sidebar.number_input("压力 (MPa)", value=1.0, step=0.1)

mode = st.sidebar.radio("优化模式", ("target", "max"), format_func=lambda x: "目标吸附容量" if x=="target" else "最大化吸附容量")
if mode == "target":
    Q_target = st.sidebar.number_input("目标吸附容量 (mmol/g)", value=3.5, step=0.1)
else:
    Q_target = None

top_n = st.sidebar.slider("输出候选数量", min_value=1, max_value=10, value=5)
run_opt = st.sidebar.button("开始优化")

# ========== 运行优化 ==========
if run_opt:
    with st.spinner("正在运行遗传算法优化，请稍候..."):
        # 提取优化所需参数
        n_cont = len(data_info['numeric_cols'])
        n_cat = len(data_info['cat_vars'])
        n_dim = n_cont + n_cat

        lb = [data_info['numeric_bounds'][col][0] for col in data_info['numeric_cols']] + [0]*n_cat
        ub = [data_info['numeric_bounds'][col][1] for col in data_info['numeric_cols']] + [len(data_info['cat_vars'][var])-0.5 for var in data_info['cat_vars']]

        # 创建遗传算法对象
        ga = GA(func=lambda x: objective_func(x, T_target, P_target, Q_target if mode=='target' else None, mode,
                                                data_info, model, data_info['scaler']),
                n_dim=n_dim,
                size_pop=100,
                max_iter=200,
                lb=lb,
                ub=ub,
                precision=1e-7)

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
            for j, (var, cats) in enumerate(data_info['cat_vars'].items()):
                if cat_indices[j] >= len(cats):
                    cat_indices[j] = len(cats) - 1

            result = {}
            for i, col in enumerate(data_info['numeric_cols']):
                result[col] = cont_vals[i]
            for j, (var, cats) in enumerate(data_info['cat_vars'].items()):
                result[var] = cats[cat_indices[j]]

            # 验证真实预测值（不含惩罚）
            x_vec = np.zeros(len(data_info['all_cols']))
            for i, col in enumerate(data_info['numeric_cols']):
                idx = data_info['all_cols'].index(col)
                x_vec[idx] = cont_vals[i]
            idx_p = data_info['all_cols'].index('Pressure_MPa')
            idx_t = data_info['all_cols'].index('Temperature_K')
            x_vec[idx_p] = P_target
            x_vec[idx_t] = T_target
            for j, (var, cats) in enumerate(data_info['cat_vars'].items()):
                cat_name = cats[cat_indices[j]]
                onehot_col = f"{var}_{cat_name}"
                if onehot_col in data_info['all_cols']:
                    idx = data_info['all_cols'].index(onehot_col)
                    x_vec[idx] = 1.0
            x_scaled = data_info['scaler'].transform(x_vec.reshape(1, -1))
            pred = model.predict(x_scaled)[0]

            candidates.append((result, pred, y))

        # 存储到会话状态以便展示
        st.session_state['candidates'] = candidates
        st.session_state['mode'] = mode
        st.session_state['Q_target'] = Q_target if mode=='target' else None

# ========== 显示结果 ==========
if 'candidates' in st.session_state:
    candidates = st.session_state['candidates']
    mode = st.session_state['mode']
    Q_target = st.session_state['Q_target']

    st.header("优化结果")
    st.subheader(f"前 {len(candidates)} 组最优候选解")

    # 将候选解转为 DataFrame 用于表格和绘图
    df_candidates = pd.DataFrame([r[0] for r in candidates])
    df_candidates['Predicted_Adsorption'] = [r[1] for r in candidates]
    df_candidates['Objective'] = [r[2] for r in candidates]

    # 显示表格
    st.dataframe(df_candidates.style.format({col: "{:.4f}" for col in data_info['numeric_cols'] + ['Predicted_Adsorption', 'Objective']}))

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
    fig1 = plt.figure(figsize=(10, 5))
    parallel_vars = data_info['numeric_cols'] + ['Predicted_Adsorption']
    pd.plotting.parallel_coordinates(df_candidates[parallel_vars], 'Predicted_Adsorption',
                                      colormap='viridis', linewidth=2, alpha=0.8)
    plt.title('Parallel Coordinates of Top Candidates')
    plt.xlabel('Variables')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=45)
    plt.legend(title='Predicted (mmol/g)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig1)

    # 2. 散点图矩阵
    fig2 = plt.figure(figsize=(10, 8))
    g = sns.pairplot(df_candidates, vars=data_info['numeric_cols']+['Predicted_Adsorption'],
                     diag_kind='hist', plot_kws={'alpha':0.8, 's':60},
                     diag_kws={'alpha':0.7, 'bins':10})
    g.fig.suptitle('Scatter Matrix of Continuous Variables', y=1.02, fontsize=16)
    st.pyplot(g.fig)

    # 3. 分类变量分布
    cat_vars_names = list(data_info['cat_vars'].keys())
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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontsize=10)
    for j in range(len(cat_vars_names), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig3)

    # 4. 预测值与目标对比（如果模式是target）
    if mode == 'target':
        fig4, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([Q_target]*len(df_candidates), df_candidates['Predicted_Adsorption'],
                   s=120, c='red', alpha=0.7, edgecolor='black', zorder=5)
        ax.plot([Q_target-0.2, Q_target+0.2], [Q_target-0.2, Q_target+0.2],
                'k--', linewidth=2, label='Perfect match')
        ax.set_xlabel('Target Adsorption (mmol/g)')
        ax.set_ylabel('Predicted Adsorption (mmol/g)')
        ax.set_title('Predicted vs Target')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        fig4, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(1, len(df_candidates)+1), df_candidates['Predicted_Adsorption'],
               color='steelblue', edgecolor='black')
        ax.set_xlabel('Candidate Rank')
        ax.set_ylabel('Predicted Adsorption (mmol/g)')
        ax.set_title('Top Candidates - Predicted Adsorption')
        ax.set_xticks(range(1, len(df_candidates)+1))
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig4)

else:
    st.info("请在侧边栏设置参数并点击「开始优化」按钮。")