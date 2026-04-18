# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sko.GA import GA
import joblib
import warnings
import time
import io  # 新增：用于 CSV 下载

warnings.filterwarnings('ignore')

# ========== 设置页面 ==========
st.set_page_config(page_title="Mg-MOF-74 Goal-oriented Design Platform", layout="wide")
st.title("Mg-MOF-74 Adsorbent Goal-oriented Design Platform")
st.markdown("Based on Bag_ET_GB_RF_XGB model and genetic algorithm to search for optimal synthesis and structural parameters for CO₂ capture.")

# ========== 英文标签映射 ==========
plot_label_mapping = {
    'Molar ratio': 'Mg/ligand ratio',
    'SBET_m2_g': r'S$_{BET}$ (m$^2$/g)',
    'Vpore_cm3_g': r'V$_{pore}$ (cm$^3$/g)',
    'dpore': r'd$_{pore}$ (nm)',
    'Pressure_MPa': 'Pressure (MPa)',
    'Temperature_K': 'Temperature (K)',
    'Adsorption_capacity_mmol_g': r'CO$_2$ uptake (mmol/g)',
    'Mg_source_acetate_mg_source': 'Acetate',
    'Mg_source_nitrate_mg_source': 'Nitrate',
    'Mg_source_other_mg_salts': 'Other Mg salts',
    'Mg_source_oxide_mg_source': 'MgO',
    'Solvent_aqueous_dmf_system': 'Aqueous DMF',
    'Solvent_non_aqueous_alcohol_dmf_system': 'Alcohol-DMF',
    'Solvent_special_two_phase_system': 'Two-phase system',
    'Treatment_carbon_treatment': 'Carbon modification',
    'Treatment_framework_nitrogen': 'Framework nitrogen',
    'Treatment_metal_treatment': 'Metal modification',
    'Treatment_no_treatment': 'No modification',
    'Treatment_organic_acid_treatment': 'Organic acid',
    'Treatment_other_treatment': 'Other modification',
    'Treatment_polymer_treatment': 'Polymer modification',
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
    model = joblib.load('Bag_ET_GB_RF_XGB.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ========== 从数据集提取特征信息 ==========
@st.cache_data
def load_feature_info():
    df = pd.read_csv('Mg_MOF_74.csv')
    train_data = df[df['split'] == 'train'].copy()
    test_data = df[df['split'] == 'test'].copy()
    target_col = 'Adsorption_capacity_mmol_g'
    
    X_train = train_data.drop(columns=[target_col, 'split'])
    y_train = train_data[target_col]
    
    numeric_cols = ['Molar ratio', 'SBET_m2_g', 'Vpore_cm3_g', 'dpore']
    fixed_cols = ['Pressure_MPa', 'Temperature_K']
    cat_vars_names = ['Mg_source', 'Solvent', 'Treatment', 'Morphology']
    all_cols = X_train.columns.tolist()
    
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
    
    # 单变量边界
    numeric_bounds = {}
    for col in numeric_cols:
        numeric_bounds[col] = (X_train[col].min(), X_train[col].max())
    
    # SBET/Vpore 比值约束
    ratio_train = X_train['SBET_m2_g'] / (X_train['Vpore_cm3_g'] + 1e-6)
    q_low = ratio_train.quantile(0.025)
    q_high = ratio_train.quantile(0.975)
    
    # 分组约束边界
    def get_cat_value(row, var_name, cat_vars_dict):
        for cat in cat_vars_dict[var_name]:
            col_name = f"{var_name}_{cat}"
            if col_name in row and row[col_name] == 1:
                return cat
        return None
    
    group_bounds = {}
    for idx, row in X_train.iterrows():
        mg = get_cat_value(row, 'Mg_source', cat_vars)
        sol = get_cat_value(row, 'Solvent', cat_vars)
        treat = get_cat_value(row, 'Treatment', cat_vars)
        morph = get_cat_value(row, 'Morphology', cat_vars)
        if None in (mg, sol, treat, morph):
            continue
        key = (mg, sol, treat, morph)
        sbet = row['SBET_m2_g']
        vpore = row['Vpore_cm3_g']
        dpore = row['dpore']
        if key not in group_bounds:
            group_bounds[key] = {
                'SBET': [sbet, sbet],
                'Vpore': [vpore, vpore],
                'dpore': [dpore, dpore]
            }
        else:
            group_bounds[key]['SBET'][0] = min(group_bounds[key]['SBET'][0], sbet)
            group_bounds[key]['SBET'][1] = max(group_bounds[key]['SBET'][1], sbet)
            group_bounds[key]['Vpore'][0] = min(group_bounds[key]['Vpore'][0], vpore)
            group_bounds[key]['Vpore'][1] = max(group_bounds[key]['Vpore'][1], vpore)
            group_bounds[key]['dpore'][0] = min(group_bounds[key]['dpore'][0], dpore)
            group_bounds[key]['dpore'][1] = max(group_bounds[key]['dpore'][1], dpore)
    
    return {
        'numeric_cols': numeric_cols,
        'fixed_cols': fixed_cols,
        'cat_vars': cat_vars,
        'all_cols': all_cols,
        'numeric_bounds': numeric_bounds,
        'ratio_bounds': (q_low, q_high),
        'group_bounds': group_bounds,
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
group_bounds = info['group_bounds']

# ========== 原有约束：SBET/Vpore 比值惩罚 ==========
def ratio_penalty(cont_vals):
    sbet = cont_vals[1]
    vpore = cont_vals[2]
    ratio = sbet / (vpore + 1e-6)
    q_low, q_high = ratio_bounds
    penalty = 0.0
    if ratio < q_low:
        penalty = 1000 * (q_low - ratio)
    elif ratio > q_high:
        penalty = 1000 * (ratio - q_high)
    return penalty

# ===== 分组约束惩罚 =====
def group_constraint_penalty(cont_vals, cat_names, penalty_weight=100.0):
    key = tuple(cat_names)
    if key not in group_bounds:
        return 0.0
    bounds = group_bounds[key]
    penalty = 0.0
    sbet = cont_vals[1]
    sbet_min, sbet_max = bounds['SBET']
    if sbet < sbet_min:
        rel = (sbet_min - sbet) / (sbet_min + 1e-6)
        penalty += penalty_weight * rel
    elif sbet > sbet_max:
        rel = (sbet - sbet_max) / (sbet_max + 1e-6)
        penalty += penalty_weight * rel
    vpore = cont_vals[2]
    vpore_min, vpore_max = bounds['Vpore']
    if vpore < vpore_min:
        rel = (vpore_min - vpore) / (vpore_min + 1e-6)
        penalty += penalty_weight * rel
    elif vpore > vpore_max:
        rel = (vpore - vpore_max) / (vpore_max + 1e-6)
        penalty += penalty_weight * rel
    dpore = cont_vals[3]
    dpore_min, dpore_max = bounds['dpore']
    if dpore < dpore_min:
        rel = (dpore_min - dpore) / (dpore_min + 1e-6)
        penalty += penalty_weight * rel
    elif dpore > dpore_max:
        rel = (dpore - dpore_max) / (dpore_max + 1e-6)
        penalty += penalty_weight * rel
    return penalty

def objective_func(x, T_target, P_target, Q_target, mode, group_penalty_weight):
    n_cont = len(numeric_cols)
    cont_vals = x[:n_cont]
    cat_indices = x[n_cont:].astype(int)

    cat_names = []
    for j, (var, cats) in enumerate(cat_vars.items()):
        if cat_indices[j] < 0 or cat_indices[j] >= len(cats):
            cat_indices[j] = np.clip(cat_indices[j], 0, len(cats)-1)
        cat_names.append(cats[cat_indices[j]])

    x_vec = np.zeros(len(all_cols))
    for i, col in enumerate(numeric_cols):
        idx = all_cols.index(col)
        x_vec[idx] = cont_vals[i]
    idx_p = all_cols.index('Pressure_MPa')
    idx_t = all_cols.index('Temperature_K')
    x_vec[idx_p] = P_target
    x_vec[idx_t] = T_target
    for j, (var, cats) in enumerate(cat_vars.items()):
        cat_name = cat_names[j]
        onehot_col = f"{var}_{cat_name}"
        if onehot_col in all_cols:
            idx = all_cols.index(onehot_col)
            x_vec[idx] = 1.0

    x_scaled = scaler.transform(x_vec.reshape(1, -1))
    pred = model.predict(x_scaled)[0]

    penalty_ratio = ratio_penalty(cont_vals)
    penalty_group = group_constraint_penalty(cont_vals, cat_names, group_penalty_weight)

    if mode == 'target':
        return abs(pred - Q_target) + penalty_ratio + penalty_group
    else:
        return -pred + penalty_ratio + penalty_group

# ===== 新增：根据候选字典和压力、温度构造特征向量并预测 =====
def predict_from_candidate(candidate_dict, pressure_mpa, temperature_k):
    """candidate_dict 包含所有特征（连续+分类），返回模型预测值"""
    x_vec = np.zeros(len(all_cols))
    # 连续变量
    for col in numeric_cols:
        idx = all_cols.index(col)
        x_vec[idx] = candidate_dict[col]
    # 固定温度压力
    idx_p = all_cols.index('Pressure_MPa')
    idx_t = all_cols.index('Temperature_K')
    x_vec[idx_p] = pressure_mpa
    x_vec[idx_t] = temperature_k
    # 分类变量（独热）
    for var in cat_vars:
        cat_name = candidate_dict[var]
        onehot_col = f"{var}_{cat_name}"
        if onehot_col in all_cols:
            idx = all_cols.index(onehot_col)
            x_vec[idx] = 1.0
    x_scaled = scaler.transform(x_vec.reshape(1, -1))
    return model.predict(x_scaled)[0]

# ===== 新增：生成等温线数据 =====
def generate_isotherm(candidate_dict, temp_k, pressure_points):
    """pressure_points: list of pressure in MPa, 返回对应预测吸附量的列表"""
    preds = []
    for p in pressure_points:
        pred = predict_from_candidate(candidate_dict, p, temp_k)
        preds.append(pred)
    return preds


# ========== 侧边栏输入 ==========
st.sidebar.header("Environment Conditions")
T_target = st.sidebar.number_input("Temperature (K)", value=298.0, min_value=273.0, max_value=333.0, step=1.0)
P_target = st.sidebar.number_input("Pressure (MPa)", value=0.1, min_value=0.0001, max_value=2.9491, step=0.1, format="%.4f")

mode = st.sidebar.radio("Optimization Mode", ("target", "max"), format_func=lambda x: "Target uptake" if x == "target" else "Maximize uptake")
if mode == "target":
    Q_target = st.sidebar.number_input("Target uptake (mmol/g)", value=3.5, step=0.1)
else:
    Q_target = None

top_n = st.sidebar.slider("Number of candidates", min_value=1, max_value=10, value=5)
use_clustering = st.sidebar.checkbox("Use clustering to select diverse candidates", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("GA Parameters")
pop_size = st.sidebar.slider("Population size", min_value=50, max_value=300, value=100, step=10,
                             help="Number of individuals in each generation. Larger populations explore more broadly but increase runtime.")
max_iter = st.sidebar.slider("Max iterations", min_value=50, max_value=500, value=150, step=10,
                             help="Maximum number of generations. Higher values allow more thorough convergence but take longer.")

group_penalty_weight = st.sidebar.slider("Group constraint penalty weight", min_value=0.0, max_value=500.0, value=100.0, step=10.0,
                                         help="Penalty strength. Larger values force structural properties to stay within the boundaries of historically successful combinations. Set to 0 to disable this constraint.")
st.sidebar.markdown("---")

run_opt = st.sidebar.button("Start Optimization")

# ========== 运行优化 ==========
if run_opt:
    with st.spinner("Running genetic algorithm optimization... This may take 30-60 seconds."):
        start_time = time.time()
        
        n_cont = len(numeric_cols)
        n_cat = len(cat_vars)
        n_dim = n_cont + n_cat

        lb = [numeric_bounds[col][0] for col in numeric_cols] + [0] * n_cat
        ub = [numeric_bounds[col][1] for col in numeric_cols] + [len(cat_vars[var]) - 0.5 for var in cat_vars]

        ga = GA(
            func=lambda x: objective_func(x, T_target, P_target, Q_target, mode, group_penalty_weight),
            n_dim=n_dim,
            size_pop=pop_size,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            precision=1e-3
        )

        best_x, best_y = ga.run()
        final_X = ga.X
        final_Y = ga.Y

        # 选择代表解
        if use_clustering and len(final_X) >= top_n:
            scaler4cluster = StandardScaler()
            X_cont = final_X[:, :n_cont]
            X_cont_scaled = scaler4cluster.fit_transform(X_cont)
            X_cat = final_X[:, n_cont:].astype(int)
            X_cluster = np.hstack([X_cont_scaled, X_cat * 0.5])
            n_clusters = min(top_n, len(final_X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
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
            if len(representative_idx) < top_n:
                additional = [i for i in np.argsort(final_Y)[:top_n] if i not in representative_idx]
                representative_idx.extend(additional[:top_n - len(representative_idx)])
            top_idx = representative_idx[:top_n]
        else:
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

        elapsed = time.time() - start_time
        st.success(f"Optimization completed in {elapsed:.1f} seconds.")
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

    df_candidates = pd.DataFrame([r[0] for r in candidates])
    df_candidates['Predicted_Adsorption'] = [r[1] for r in candidates]
    df_candidates['Objective'] = [r[2] for r in candidates]

    df_display = df_candidates.copy()
    cat_cols = ['Mg_source', 'Solvent', 'Treatment', 'Morphology']
    for col in cat_cols:
        df_display[col] = df_display[col].apply(lambda x: plot_label_mapping.get(f'{col}_{x}', x))
    df_display = df_display.rename(columns={'Treatment': 'Modification'})

    st.dataframe(
        df_display.style.format(
            {col: "{:.4f}" for col in numeric_cols + ['Predicted_Adsorption', 'Objective']}
        )
    )

    # CSV下载
    df_csv = df_candidates.copy()
    for col in cat_cols:
        df_csv[col] = df_csv[col].apply(lambda x: plot_label_mapping.get(f'{col}_{x}', x))
    df_csv = df_csv.rename(columns={'Treatment': 'Modification'})
    csv = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button("Download candidates as CSV", data=csv, file_name="candidates.csv", mime="text/csv")

    # 可视化部分（原六子图）
    st.markdown("### Visualizations")
    df_viz = df_candidates.copy()
    n_candidates = len(df_viz)
    cmap = plt.cm.Set3_r
    norm = plt.Normalize(df_viz['Predicted_Adsorption'].min(), df_viz['Predicted_Adsorption'].max())
    colors_ads = [cmap(norm(val)) for val in df_viz['Predicted_Adsorption']]

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.08, right=0.92, bottom=0.1, top=0.92)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # (a) SBET vs Vpore
    sc_a = ax_a.scatter(df_viz['SBET_m2_g'], df_viz['Vpore_cm3_g'],
                        c=df_viz['Predicted_Adsorption'], cmap='Set3_r',
                        s=100, edgecolor='k', linewidth=0.8, zorder=5)
    ax_a.set_xlabel(plot_label_mapping['SBET_m2_g'], fontsize=13, fontweight='bold')
    ax_a.set_ylabel(plot_label_mapping['Vpore_cm3_g'], fontsize=13, fontweight='bold')
    ax_a.set_title('(a) Surface area vs pore volume', fontsize=15, fontweight='bold')
    ax_a.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.setp(ax_a.get_xticklabels(), fontweight='bold')
    plt.setp(ax_a.get_yticklabels(), fontweight='bold')
    cbar_a = plt.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label(r'CO$_2$ uptake (mmol/g)', fontsize=12, fontweight='bold')
    plt.setp(cbar_a.ax.get_yticklabels(), fontweight='bold')

    # (b) dpore vs Adsorption
    sc_b = ax_b.scatter(df_viz['dpore'], df_viz['Predicted_Adsorption'],
                        c=df_viz['Molar ratio'], cmap='Set3_r',
                        s=100, edgecolor='k', linewidth=0.8, zorder=5)
    ax_b.set_xlabel(r'd$_{\text{pore}}$ (nm)', fontsize=13, fontweight='bold')
    ax_b.set_ylabel(r'CO$_2$ uptake (mmol/g)', fontsize=13, fontweight='bold')
    ax_b.set_title('(b) Pore size vs uptake', fontsize=15, fontweight='bold')
    ax_b.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.setp(ax_b.get_xticklabels(), fontweight='bold')
    plt.setp(ax_b.get_yticklabels(), fontweight='bold')
    cbar_b = plt.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label('Mg/ligand ratio', fontsize=12, fontweight='bold')
    plt.setp(cbar_b.ax.get_yticklabels(), fontweight='bold')

    # (c) Molar ratio vs Adsorption
    morph_categories = df_viz['Morphology'].astype('category')
    morph_codes = morph_categories.cat.codes
    unique_morphs = morph_categories.cat.categories
    cmap_morph = plt.cm.tab10
    sc_c = ax_c.scatter(df_viz['Molar ratio'], df_viz['Predicted_Adsorption'],
                        c=morph_codes, cmap='tab10', s=100, edgecolor='k', linewidth=0.8, zorder=5)
    ax_c.set_xlabel(plot_label_mapping['Molar ratio'], fontsize=13, fontweight='bold')
    ax_c.set_ylabel(r'CO$_2$ uptake (mmol/g)', fontsize=13, fontweight='bold')
    ax_c.set_title('(c) Molar ratio vs uptake', fontsize=15, fontweight='bold')
    ax_c.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.setp(ax_c.get_xticklabels(), fontweight='bold')
    plt.setp(ax_c.get_yticklabels(), fontweight='bold')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_morph(i), markersize=8)
               for i in range(len(unique_morphs))]
    legend = ax_c.legend(handles, [plot_label_mapping.get('Morphology_'+m, m) for m in unique_morphs],
                         title='Morphology', fontsize=9, loc='upper left', bbox_to_anchor=(1,1),
                         prop={'weight':'bold'})
    plt.setp(legend.get_title(), fontweight='bold')

    # (d) 合成条件分布
    gs_d = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1,0], hspace=0.35, wspace=0.35)
    axes_d = [fig.add_subplot(gs_d[i, j]) for i in range(2) for j in range(2)]
    cat_vars_display = ['Mg_source', 'Solvent', 'Treatment', 'Morphology']
    titles_d = ['(d1) Mg source', '(d2) Solvent', '(d3) Modification', '(d4) Morphology']
    for idx, (var, ax, title) in enumerate(zip(cat_vars_display, axes_d, titles_d)):
        counts = df_viz[var].value_counts()
        cats = [plot_label_mapping.get(f'{var}_{c}', c) for c in counts.index]
        x_pos = np.arange(len(cats))
        colors_d = [plt.cm.Set3_r(i / max(1, len(counts)-1)) for i in range(len(counts))]
        bars = ax.bar(x_pos, counts.values, color=colors_d, edgecolor='black')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cats, rotation=0, ha='center', fontweight='bold', fontsize=12)
        plt.setp(ax.get_yticklabels(), fontweight='bold')
        max_count = counts.max()
        ax.set_ylim(0, max_count * 1.2)
        for bar, cnt in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(cnt),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_d.axis('off')

    # (e) 性能排序条形图
    x_pos = np.arange(1, n_candidates+1)
    bars_e = ax_e.bar(x_pos, df_viz['Predicted_Adsorption'], color=colors_ads, edgecolor='black')
    ax_e.set_xlabel('Candidate', fontsize=13, fontweight='bold')
    ax_e.set_ylabel(r'CO$_2$ uptake (mmol/g)', fontsize=13, fontweight='bold')
    ax_e.set_title('(e) Adsorption capacity of candidates', fontsize=15, fontweight='bold')
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels([f'{i}' for i in x_pos], fontweight='bold')
    ax_e.grid(axis='y', linestyle='--', alpha=0.3)
    plt.setp(ax_e.get_yticklabels(), fontweight='bold')
    for bar, val in zip(bars_e, df_viz['Predicted_Adsorption']):
        ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                  ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (f) 平行坐标图
    cont_all = ['Molar ratio', 'SBET_m2_g', 'Vpore_cm3_g', 'dpore', 'Predicted_Adsorption']
    scaler_parallel = MinMaxScaler()
    data_parallel = scaler_parallel.fit_transform(df_viz[cont_all])
    x_ticks = np.arange(len(cont_all))
    for i in range(n_candidates):
        ax_f.plot(x_ticks, data_parallel[i], color=colors_ads[i], linewidth=2, alpha=0.8, marker='o', markersize=6)
    ax_f.set_xticks(x_ticks)
    ax_f.set_xticklabels([plot_label_mapping.get(c, c) for c in cont_all], rotation=45, ha='right', fontsize=11, weight='bold')
    ax_f.set_ylabel('Normalized value', fontsize=13, fontweight='bold')
    ax_f.set_title('(f) Parallel coordinates of all variables', fontsize=15, fontweight='bold')
    ax_f.grid(True, linestyle='--', alpha=0.3)
    plt.setp(ax_f.get_yticklabels(), fontweight='bold')
    norm_f = plt.Normalize(df_viz['Predicted_Adsorption'].min(), df_viz['Predicted_Adsorption'].max())
    sm_f = plt.cm.ScalarMappable(cmap='Set3_r', norm=norm_f)
    sm_f.set_array([])
    cbar_f = fig.colorbar(sm_f, ax=ax_f, fraction=0.046, pad=0.04)
    cbar_f.set_label(r'CO$_2$ uptake (mmol/g)', fontsize=12, fontweight='bold')
    plt.setp(cbar_f.ax.get_yticklabels(), fontweight='bold')

    plt.suptitle(r'Goal-oriented design of Mg-MOF-74 for CO$_2$ capture', fontsize=24, weight='bold', y=0.98)
    st.pyplot(fig)

    # ========== 新增：等温线预测功能 ==========
    st.markdown("---")
    st.subheader("📈 Predict adsorption isotherm for a candidate")

    # 让用户选择候选解（索引从1开始）
    candidate_indices = list(range(1, len(candidates) + 1))
    selected_idx = st.selectbox("Select candidate number", candidate_indices, index=0)
    selected_candidate_dict = candidates[selected_idx - 1][0]  # 字典

    # 压力点设置（MPa），默认从0.001到0.1，共15个点（可自定义）
    st.markdown("**Pressure range (MPa)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        p_min = st.number_input("Min pressure", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")
    with col2:
        p_max = st.number_input("Max pressure", value=0.1, min_value=0.01, max_value=2.0, format="%.4f")
    with col3:
        num_points = st.slider("Number of points", min_value=5, max_value=30, value=15)

    # 固定温度使用当前侧边栏的温度（或允许用户临时修改）
    use_current_temp = st.checkbox("Use current temperature from sidebar", value=True)
    if use_current_temp:
        isotherm_temp = T_target
    else:
        isotherm_temp = st.number_input("Temperature for isotherm (K)", value=298.0, min_value=273.0, max_value=333.0)

    if st.button("Generate isotherm"):
        with st.spinner("Generating isotherm..."):
            pressures = np.linspace(p_min, p_max, num_points)
            preds = generate_isotherm(selected_candidate_dict, isotherm_temp, pressures)

            # 绘制等温线图
            fig_iso, ax_iso = plt.subplots(figsize=(8, 5))
            ax_iso.plot(pressures, preds, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Model prediction')
            ax_iso.set_xlabel('Pressure (MPa)', fontsize=12, fontweight='bold')
            ax_iso.set_ylabel(r'CO$_2$ uptake (mmol/g)', fontsize=12, fontweight='bold')
            ax_iso.set_title(f'Predicted adsorption isotherm at {isotherm_temp} K\nCandidate {selected_idx}', fontsize=14, fontweight='bold')
            ax_iso.grid(True, linestyle='--', alpha=0.3)
            ax_iso.legend()
            st.pyplot(fig_iso)

            # 提供数据下载
            iso_df = pd.DataFrame({'Pressure_MPa': pressures, 'Predicted_uptake_mmol_g': preds})
            csv_iso = iso_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download isotherm data as CSV", data=csv_iso, file_name=f"isotherm_candidate_{selected_idx}.csv", mime="text/csv")

            # 可选：显示预测值表格
            with st.expander("Show predicted values"):
                st.dataframe(iso_df.style.format({'Pressure_MPa': '{:.4f}', 'Predicted_uptake_mmol_g': '{:.3f}'}))
    # ========================================

else:
    st.info("Please set parameters in the sidebar and click 'Start Optimization'.")
