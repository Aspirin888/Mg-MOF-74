# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from sko.GA import GA
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 设置页面 ==========
st.set_page_config(page_title="Mg-MOF-74 Inverse Design Platform", layout="wide")
st.title("Mg-MOF-74 Adsorbent Inverse Design Platform")
st.markdown("Based on CatBoost model and genetic algorithm to search for optimal synthesis and structural parameters for CO₂ capture.")

# ========== 英文标签映射 ==========
plot_label_mapping = {
    'Molar ratio': 'Mg/ligand ratio',
    'SBET_m2_g': r'S$_{BET}$ (m$^2$/g)',
    'Vpore_cm3_g': r'V$_{pore}$ (cm$^3$/g)',
    'Pressure_MPa': 'Pressure (MPa)',
    'Temperature_K': 'Temperature (K)',
    'Adsorption_capacity_mmol_g': r'CO$_2$ uptake (mmol/g)',
    # 合成分类变量
    'Mg_source_acetate_mg_source': 'Acetate',
    'Mg_source_nitrate_mg_source': 'Nitrate',
    'Mg_source_other_mg_salts': 'Other Mg salts',
    'Mg_source_oxide_mg_source': 'MgO',
    'Solvent_aqueous_dmf_system': 'Aqueous DMF',
    'Solvent_non_aqueous_alcohol_dmf_system': 'Alcohol-DMF',
    'Solvent_special_two_phase_system': 'Two-phase system',
    'Treatment_carbon_treatment': 'Carbon',
    'Treatment_framework_nitrogen': 'Framework nitrogen',
    'Treatment_metal_treatment': 'Metal',
    'Treatment_no_treatment': 'No treatment',
    'Treatment_organic_acid_treatment': 'Organic acid',
    'Treatment_other_treatment': 'Other treatment',
    'Treatment_polymer_treatment': 'Polymer',
    'Treatment_post_base_treatment': 'Post base',
    'Morphology_flower_like_structure': 'Flower-like',
    'Morphology_other_morphology': 'Other',
    'Morphology_porous_structure': 'Porous',
    'Morphology_rod_columnar_structure': 'Rod/columnar',
    'Morphology_spherical_structure': 'Spherical'
}

# ========== 加载模型和标准化器 ==========
@st.cache_resource
def load_model():
    model = joblib.load('CatBoost.pkl')        # 确保模型文件存在
    scaler = joblib.load('scaler.pkl')          # 确保标准化器文件存在
    return model, scaler

model, scaler = load_model()

# ========== 从数据集提取特征信息 ==========
@st.cache_data
def load_feature_info():
    df = pd.read_csv('Mg_MOF_74_独热编码.csv')
    X = df.drop('Adsorption_capacity_mmol_g', axis=1)
    y = df['Adsorption_capacity_mmol_g']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_cols = ['Molar ratio', 'SBET_m2_g', 'Vpore_cm3_g']
    fixed_cols = ['Pressure_MPa', 'Temperature_K']
    cat_vars_names = ['Mg_source', 'Solvent', 'Treatment', 'Morphology']
    all_cols = X.columns.tolist()
    
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
    
    numeric_bounds = {}
    for col in numeric_cols:
        numeric_bounds[col] = (X_train[col].min(), X_train[col].max())
    
    ratio_train = X_train['SBET_m2_g'] / (X_train['Vpore_cm3_g'] + 1e-6)
    q_low = ratio_train.quantile(0.025)
    q_high = ratio_train.quantile(0.975)
    
    return {
        'numeric_cols': numeric_cols,
        'fixed_cols': fixed_cols,
        'cat_vars': cat_vars,
        'all_cols': all_cols,
        'numeric_bounds': numeric_bounds,
        'ratio_bounds': (q_low, q_high),
        'X_train': X_train,
        'y_train': y_train
    }

info = load_feature_info()
numeric_cols = info['numeric_cols']
fixed_cols = info['fixed_cols']
cat_vars = info['cat_vars']
all_cols = info['all_cols']
numeric_bounds = info['numeric_bounds']
ratio_bounds = info['ratio_bounds']

# ========== 约束和目标函数 ==========
def constraint_penalty(x):
    sbet = x[1]      # SBET_m2_g is the second continuous variable
    vpore = x[2]     # Vpore_cm3_g is the third
    ratio = sbet / (vpore + 1e-6)
    q_low, q_high = ratio_bounds
    penalty = 0.0
    if ratio < q_low:
        penalty = 1000 * (q_low - ratio)
    elif ratio > q_high:
        penalty = 1000 * (ratio - q_high)
    return penalty

def objective_func(x, T_target, P_target, Q_target, mode):
    n_cont = len(numeric_cols)
    cont_vals = x[:n_cont]
    cat_indices = x[n_cont:].astype(int)

    for j, (var, cats) in enumerate(cat_vars.items()):
        if cat_indices[j] < 0 or cat_indices[j] >= len(cats):
            cat_indices[j] = np.clip(cat_indices[j], 0, len(cats)-1)

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

    penalty = constraint_penalty(cont_vals)

    if mode == 'target':
        return abs(pred - Q_target) + penalty
    else:  # maximize
        return -pred + penalty

# ========== 侧边栏输入 ==========
st.sidebar.header("Environment Conditions")
T_target = st.sidebar.number_input("Temperature (K)", value=298.0, step=1.0)
P_target = st.sidebar.number_input("Pressure (MPa)", value=1.0, step=0.1)

mode = st.sidebar.radio(
    "Optimization Mode",
    ("target", "max"),
    format_func=lambda x: "Target uptake" if x == "target" else "Maximize uptake"
)
if mode == "target":
    Q_target = st.sidebar.number_input("Target uptake (mmol/g)", value=3.5, step=0.1)
else:
    Q_target = None

top_n = st.sidebar.slider("Number of candidates", min_value=1, max_value=10, value=5)
use_clustering = st.sidebar.checkbox("Use clustering to select diverse candidates", value=True)
run_opt = st.sidebar.button("Start Optimization")

# ========== 运行优化 ==========
if run_opt:
    with st.spinner("Running genetic algorithm optimization..."):
        n_cont = len(numeric_cols)
        n_cat = len(cat_vars)
        n_dim = n_cont + n_cat

        lb = [numeric_bounds[col][0] for col in numeric_cols] + [0] * n_cat
        ub = [numeric_bounds[col][1] for col in numeric_cols] + [len(cat_vars[var]) - 0.5 for var in cat_vars]

        ga = GA(
            func=lambda x: objective_func(x, T_target, P_target, Q_target, mode),
            n_dim=n_dim,
            size_pop=200,
            max_iter=300,
            lb=lb,
            ub=ub,
            precision=1e-7
        )

        best_x, best_y = ga.run()

        # 获取所有最终种群个体
        final_X = ga.X
        final_Y = ga.Y

        # 选择代表解
        if use_clustering and len(final_X) >= top_n:
            # KMeans聚类选择
            scaler4cluster = StandardScaler()
            X_cont = final_X[:, :n_cont]
            X_cont_scaled = scaler4cluster.fit_transform(X_cont)
            X_cat = final_X[:, n_cont:].astype(int)
            X_cluster = np.hstack([X_cont_scaled, X_cat * 0.5])

            n_clusters = min(top_n, len(final_X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_cluster)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            representative_idx = []
            for i in range(n_clusters):
                cluster_points = X_cluster[labels == i]
                if len(cluster_points) == 0:
                    continue
                center = centers[i]
                dist = np.linalg.norm(cluster_points - center, axis=1)
                nearest_idx = np.where(labels == i)[0][np.argmin(dist)]
                representative_idx.append(nearest_idx)

            # 如果不够，补充目标函数最优的个体
            if len(representative_idx) < top_n:
                additional = [i for i in np.argsort(final_Y)[:top_n] if i not in representative_idx]
                representative_idx.extend(additional[:top_n - len(representative_idx)])

            top_idx = representative_idx[:top_n]
        else:
            # 简单取目标函数最优的前top_n个
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

        st.session_state['candidates'] = candidates
        st.session_state['mode'] = mode
        st.session_state['Q_target'] = Q_target if mode == 'target' else None

# ========== 显示结果 ==========
if 'candidates' in st.session_state:
    candidates = st.session_state['candidates']
    mode = st.session_state['mode']
    Q_target = st.session_state['Q_target']

    st.header("Optimization Results")
    st.subheader(f"Top {len(candidates)} candidate solutions")

    # 转换为DataFrame
    df_candidates = pd.DataFrame([r[0] for r in candidates])
    df_candidates['Predicted_Adsorption'] = [r[1] for r in candidates]
    df_candidates['Objective'] = [r[2] for r in candidates]

    # 显示表格
    st.dataframe(
        df_candidates.style.format(
            {col: "{:.4f}" for col in numeric_cols + ['Predicted_Adsorption', 'Objective']}
        )
    )

    csv = df_candidates.to_csv(index=False).encode('utf-8')
    st.download_button("Download candidates as CSV", data=csv, file_name="candidates.csv", mime="text/csv")

    # ---------- 改进的可视化（两行三列） ----------
    st.markdown("### Visualizations")

    # 准备数据
    cont_vars_structure = ['SBET_m2_g', 'Vpore_cm3_g', 'Predicted_Adsorption']
    scaler_mm = MinMaxScaler()
    data_mm_struct = scaler_mm.fit_transform(df_candidates[cont_vars_structure])
    df_mm_struct = pd.DataFrame(data_mm_struct, columns=cont_vars_structure)

    # 颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_candidates)))

    # 创建图形
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1.25, 1.25, 0.9],
                           wspace=0.4, hspace=0.5,
                           left=0.12, right=0.94, bottom=0.1, top=0.93)

    ax_a = fig.add_subplot(gs[0, 0])  # (a) 结构平行坐标
    ax_b = fig.add_subplot(gs[0, 1])  # (b) 结构-性能散点
    ax_c = fig.add_subplot(gs[0, 2])  # (c) 形貌分布
    ax_d = fig.add_subplot(gs[1, 0])  # (d) 摩尔比散点
    ax_e = fig.add_subplot(gs[1, 1])  # (e) 合成分类变量分布
    ax_f = fig.add_subplot(gs[1, 2])  # (f) 预测值排序

    # ----- (a) 结构平行坐标图 -----
    for i in range(len(df_mm_struct)):
        ax_a.plot(range(len(cont_vars_structure)), df_mm_struct.iloc[i, :],
                  color='gray', alpha=0.2, linewidth=1, zorder=1)
    for i in range(len(df_mm_struct)):
        ax_a.plot(range(len(cont_vars_structure)), df_mm_struct.iloc[i, :],
                  color=colors[i], linewidth=2, label=f'C{i+1}', zorder=2)
    xticklabels = [plot_label_mapping.get(v, v) for v in cont_vars_structure]
    xticklabels[-1] = r'CO$_2$ uptake (mmol/g)'
    ax_a.set_xticks(range(len(cont_vars_structure)))
    ax_a.set_xticklabels(xticklabels, rotation=45, ha='right', fontweight='bold', fontsize=12)
    ax_a.set_ylabel('Normalized value', fontweight='bold', fontsize=12)
    ax_a.set_title('(a) Structural features', fontweight='bold', fontsize=14)
    ax_a.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, prop={'weight':'bold', 'size':11})
    ax_a.tick_params(axis='both', labelsize=11)

    # ----- (b) 结构-性能散点图 -----
    sc = ax_b.scatter(df_candidates['SBET_m2_g'], df_candidates['Vpore_cm3_g'],
                      c=df_candidates['Predicted_Adsorption'], cmap='viridis',
                      s=120, edgecolor='k', linewidth=0.8,
                      vmin=df_candidates['Predicted_Adsorption'].min(),
                      vmax=df_candidates['Predicted_Adsorption'].max())
    ax_b.set_xlabel(plot_label_mapping['SBET_m2_g'], fontweight='bold', fontsize=12)
    ax_b.set_ylabel(plot_label_mapping['Vpore_cm3_g'], fontweight='bold', fontsize=12)
    ax_b.set_title('(b) Structure–performance map', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label(r'CO$_2$ uptake (mmol/g)', fontweight='bold', fontsize=11)
    cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.tick_params(labelsize=11)
    ax_b.tick_params(labelsize=11)

    # ----- (c) 形貌分布 -----
    morph_counts = df_candidates['Morphology'].value_counts()
    cats_readable = [plot_label_mapping.get('Morphology_' + c, c) for c in morph_counts.index]
    bars_c = ax_c.barh(cats_readable, morph_counts.values, color=colors[:len(morph_counts)], edgecolor='black')
    ax_c.set_xlabel('Count', fontweight='bold', fontsize=12)
    ax_c.set_title('(c) Morphology distribution', fontweight='bold', fontsize=14)
    for bar, count in zip(bars_c, morph_counts.values):
        ax_c.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, str(count),
                  va='center', ha='left', fontsize=12, fontweight='bold')
    ax_c.tick_params(axis='x', labelsize=11)
    ax_c.tick_params(axis='y', labelsize=11)

    # ----- (d) 摩尔比与预测值散点图 -----
    sc_d = ax_d.scatter(range(1, len(df_candidates)+1), df_candidates['Molar ratio'],
                        c=df_candidates['Predicted_Adsorption'], cmap='viridis',
                        s=120, edgecolor='k', linewidth=0.8,
                        vmin=df_candidates['Predicted_Adsorption'].min(),
                        vmax=df_candidates['Predicted_Adsorption'].max())
    ax_d.set_xlabel('Candidate', fontweight='bold', fontsize=11)
    ax_d.set_ylabel(plot_label_mapping['Molar ratio'], fontweight='bold', fontsize=12)
    ax_d.set_title('(d) Molar ratio vs candidate', fontweight='bold', fontsize=14)
    ax_d.set_xticks(range(1, len(df_candidates)+1))
    ax_d.tick_params(labelsize=11)
    cbar_d = plt.colorbar(sc_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label(r'CO$_2$ uptake (mmol/g)', fontweight='bold', fontsize=12)

    # ----- (e) 合成分类变量分布（合并）-----
    syn_cat_vars = ['Mg_source', 'Solvent', 'Treatment']
    all_cats = []
    all_counts = []
    all_colors = []
    for vi, var in enumerate(syn_cat_vars):
        counts = df_candidates[var].value_counts()
        for cat, cnt in counts.items():
            full_key = var + '_' + cat
            readable = plot_label_mapping.get(full_key, cat)
            all_cats.append(readable)
            all_counts.append(cnt)
            all_colors.append(colors[vi % len(colors)])

    bars_e = ax_e.barh(range(len(all_cats)), all_counts, color=all_colors, edgecolor='black')
    ax_e.set_yticks(range(len(all_cats)))
    ax_e.set_yticklabels(all_cats, fontweight='bold', fontsize=11)
    ax_e.set_xlabel('Count', fontweight='bold', fontsize=12)
    ax_e.set_title('(e) Synthesis categorical variables', fontweight='bold', fontsize=14)
    for bar, cnt in zip(bars_e, all_counts):
        ax_e.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(cnt),
                  va='center', ha='left', fontsize=12, fontweight='bold')
    ax_e.tick_params(axis='x', labelsize=9)

    # ----- (f) 预测值排序条形图 -----
    x_pos = np.arange(1, len(df_candidates)+1)
    bars_f = ax_f.bar(x_pos, df_candidates['Predicted_Adsorption'], color=colors, edgecolor='black')
    ax_f.set_xlabel('Candidate', fontweight='bold', fontsize=12)
    ax_f.set_ylabel(r'CO$_2$ uptake (mmol/g)', fontweight='bold', fontsize=12)
    ax_f.set_title('(f) Adsorption capacity of candidates', fontweight='bold', fontsize=14)
    ax_f.set_xticks(x_pos)
    ax_f.set_xticklabels([f'{i}' for i in x_pos], fontweight='bold', fontsize=11)
    ax_f.tick_params(axis='y', labelsize=11)

    plt.suptitle(r'Reverse design of Mg-MOF-74 for CO$_2$ capture', fontsize=22, weight='bold', y=0.98)
    st.pyplot(fig)

else:
    st.info("Please set parameters in the sidebar and click 'Start Optimization'.")

