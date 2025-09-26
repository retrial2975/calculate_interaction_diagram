import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ฟังก์ชันคำนวณต่างๆ (ไม่มีการเปลี่ยนแปลง) ---
def generate_steel_positions(b, h, nb, nh, d_prime):
    bar_positions = []
    if nb > 0:
        x_coords_b = np.linspace(d_prime, b - d_prime, nb)
        for x in x_coords_b: bar_positions.extend([(x, d_prime), (x, h - d_prime)])
    if nh > 2:
        y_coords_h = np.linspace(d_prime, h - d_prime, nh)[1:-1]
        for y in y_coords_h: bar_positions.extend([(d_prime, y), (b - d_prime, y)])
    return sorted(list(set(bar_positions)))
def get_layers_from_positions(steel_positions, axis):
    layers, coord_index = {}, 1 if axis == 'X' else 0
    for pos in steel_positions:
        layer_pos = pos[coord_index]
        layers[layer_pos] = layers.get(layer_pos, 0) + 1
    return layers
def calculate_interaction_diagram(fc, fy, b, h, layers, bar_area, column_type='Tied'):
    Es, epsilon_c_max = 2.0e6, 0.003
    epsilon_y = fy / Es
    beta1 = np.interp(fc, [0, 280, 560, np.inf], [0.85, 0.85, 0.65, 0.65])
    steel_pos = sorted(layers.keys())
    steel_areas = [layers[pos] * bar_area for pos in steel_pos]
    Ast_total, Ag, d_t = sum(steel_areas), b * h, max(steel_pos)
    Pn_nom_list, Mn_nom_list, Pn_design_list, Mn_design_list = [], [], [], []
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    phi_comp, alpha = (0.65, 0.80) if column_type == 'Tied' else (0.75, 0.85)
    phi_Pn_max_aci = alpha * phi_comp * Pn_pc
    Pn_nom_list.append(Pn_pc); Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pc * phi_comp); Mn_design_list.append(0.0)
    for c in np.logspace(np.log10(d_t / 100), np.log10(h * 5), 300):
        a = beta1 * c
        if a > h: a = h
        Cc = 0.85 * fc * a * b
        Mc = Cc * (h / 2.0 - a / 2.0)
        Pn_s, Mn_s = 0.0, 0.0
        for i, d_i in enumerate(steel_pos):
            epsilon_s = epsilon_c_max * (c - d_i) / c
            fs = np.clip(Es * epsilon_s, -fy, fy)
            force = fs * steel_areas[i] - (0.85 * fc * steel_areas[i] if fs >= 0 else 0)
            Pn_s += force
            Mn_s += force * (h / 2.0 - d_i)
        Pn, Mn = Cc + Pn_s, Mc + Mn_s
        if Mn >= 0:
            epsilon_t = epsilon_c_max * (d_t - c) / c if c > 0 else float('inf')
            phi_limits = (0.65, 0.90) if column_type == 'Tied' else (0.75, 0.90)
            phi = np.interp(epsilon_t, [epsilon_y, 0.005], phi_limits)
            phi = np.clip(phi, phi_limits[0], 0.90)
            Pn_nom_list.append(Pn); Mn_nom_list.append(Mn)
            Pn_design_list.append(Pn * phi); Mn_design_list.append(Mn * phi)
    Pn_pt = -fy * Ast_total
    Pn_nom_list.append(Pn_pt); Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pt * 0.90); Mn_design_list.append(0.0)
    Pn_nom, Mn_nom = np.array(Pn_nom_list), np.array(Mn_nom_list)
    Pn_design, Mn_design = np.array(Pn_design_list), np.array(Mn_design_list)
    sort_indices = np.argsort(Pn_nom)[::-1]
    return (Pn_nom[sort_indices]/1000, Mn_nom[sort_indices]/100000,
            Pn_design[sort_indices]/1000, Mn_design[sort_indices]/100000,
            phi_Pn_max_aci / 1000)
def calculate_euler_load(fc, b, h, beta_d, k, Lu_m):
    if pd.isna(Lu_m): return np.nan
    Ec = 15100 * np.sqrt(fc)
    Ig = (b * h**3) / 12
    EI_eff = (0.4 * Ec * Ig) / (1 + beta_d)
    Lu_cm = Lu_m * 100
    if (k * Lu_cm) == 0: return float('inf')
    return (np.pi**2 * EI_eff) / (k * Lu_cm)**2 / 1000
def get_magnified_moment_and_delta(Pu_ton, Mu_ton, Pc_ton, Cm):
    if Pu_ton <= 0 or pd.isna(Pc_ton) or pd.isna(Cm): return Mu_ton, 1.0
    denominator = 1 - (abs(Pu_ton) / (0.75 * Pc_ton))
    if denominator <= 0: return 999.0, 999.0
    delta_ns = max(1.0, Cm / denominator)
    return delta_ns * Mu_ton, delta_ns
def calculate_cm_for_group(group, moment_col):
    if len(group) < 2: return 1.0
    top_moment = group.loc[group['Station'].idxmax()][moment_col]
    bot_moment = group.loc[group['Station'].idxmin()][moment_col]
    M2, M1 = (top_moment, bot_moment) if abs(top_moment) >= abs(bot_moment) else (bot_moment, top_moment)
    if M2 == 0: return 1.0
    Cm = 0.6 + 0.4 * (M1 / M2)
    return max(0.4, min(Cm, 1.0))
def calculate_minimum_moment(Pu_ton, h_cm):
    if Pu_ton <= 0: return 0.0
    Pu_kg = abs(Pu_ton) * 1000
    M_min_kg_cm = Pu_kg * (1.5 + 0.03 * h_cm)
    return M_min_kg_cm / 100000
def draw_column_section_plotly(b, h, steel_positions, bar_dia_mm):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="Black", width=2), fillcolor="LightGrey", layer='below')
    bar_x = [pos[0] for pos in steel_positions]
    bar_y = [pos[1] for pos in steel_positions]
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers', marker=dict(color='DarkSlateGray', size=bar_dia_mm*0.5, line=dict(color='Black', width=1)), hoverinfo='none'))
    fig.update_layout(title="Column Cross-Section", yaxis_scaleanchor="x", width=500, height=500, showlegend=False)
    return fig

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator (ACI Compliant)")
with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    column_type = st.radio("ประเภทเหล็กปลอก:", ('เหล็กปลอกเดี่ยว (Tied)', 'เหล็กปลอกเกลียว (Spiral)'))
    bending_axis = st.radio("เลือกแกนคำนวณโมเมนต์:", ('X (Strong Axis)', 'Y (Weak Axis)'))
    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("fc' (ksc)", value=280.0, min_value=1.0)
        fy = st.number_input("fy (ksc)", value=4000.0, min_value=1.0)
    with st.expander("ขนาดหน้าตัด", expanded=True):
        b_in = st.number_input("ความกว้าง, b (cm)", value=40.0, min_value=1.0)
        h_in = st.number_input("ความลึก, h (cm)", value=60.0, min_value=1.0)
    with st.expander("ข้อมูลเหล็กเสริม", expanded=True):
        d_prime = st.number_input("d' (cm)", value=6.0, min_value=1.0)
        bar_dia_mm = st.selectbox("ขนาดเหล็กเสริม", [12, 16, 20, 25, 28, 32], index=3)
        nb = st.number_input("จำนวนเหล็กด้าน b (บน-ล่าง)", value=5, min_value=2)
        nh = st.number_input("จำนวนเหล็กด้าน h (ข้าง)", value=3, min_value=2)
    st.markdown("---")
    with st.expander("ข้อมูลความชะลูด & การออกแบบ", expanded=False):
        check_slenderness = st.checkbox("พิจารณาผลของความชะลูด (Slenderness)")
        k_factor = st.number_input("k-factor", value=1.0, disabled=not check_slenderness)
        beta_d = st.number_input("βd", value=0.6, disabled=not check_slenderness)
        auto_calculate_cm = st.checkbox("คำนวณ Cm Factor อัตโนมัติ", value=True, disabled=not check_slenderness)
        if not auto_calculate_cm:
            Cm_factor_manual = st.number_input("Cm Factor (Manual)", value=1.0, disabled=not check_slenderness)
        check_min_moment = st.checkbox("พิจารณา Minimum Moment ตาม ACI", value=True)
    st.markdown("---"); st.header("ตรวจสอบแรงจากไฟล์ CSV")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])
    if 'story_lu_df' not in st.session_state: st.session_state.story_lu_df = None
    df_loads = None
    if uploaded_file:
        try:
            df_loads = pd.read_csv(uploaded_file)
            required_cols = {'Story', 'Column', 'Unique Name', 'Output Case', 'Station', 'P', 'M2', 'M3'}
            if not required_cols.issubset(df_loads.columns):
                st.sidebar.error(f"ไฟล์ CSV ขาดคอลัมน์ที่จำเป็น"); df_loads = None
            else:
                column_options = sorted(df_loads['Column'].unique())
                st.write("**เลือกเสา:**"); c1, c2 = st.columns(2)
                if c1.button("เลือกทั้งหมด", key='sc'): st.session_state.selected_columns = column_options
                if c2.button("ยกเลิกทั้งหมด", key='cc'): st.session_state.selected_columns = []
                selected_columns = st.multiselect("เลือกหมายเลขเสา:", column_options, key='selected_columns')
                if selected_columns:
                    story_options = sorted(df_loads[df_loads['Column'].isin(selected_columns)]['Story'].unique())
                    st.write("**เลือกชั้น:**"); s1, s2 = st.columns(2)
                    if s1.button("เลือกทั้งหมด", key='ss'): st.session_state.selected_stories = story_options
                    if s2.button("ยกเลิกทั้งหมด", key='cs'): st.session_state.selected_stories = []
                    selected_stories = st.multiselect("เลือกชั้น:", story_options, key='selected_stories')
        except Exception as e: st.sidebar.error(f"เกิดข้อผิดพลาด: {e}")
    if check_slenderness and 'selected_stories' in st.session_state and st.session_state.selected_stories:
        st.markdown("**กรอกความสูงแต่ละชั้น (Lu):**")
        if st.session_state.story_lu_df is None or set(st.session_state.story_lu_df['Story'].astype(str)) != set([str(s) for s in st.session_state.selected_stories]):
            story_lu_data = {'Story': sorted(st.session_state.selected_stories), 'Lu (m)': [3.0] * len(st.session_state.selected_stories)}
            st.session_state.story_lu_df = pd.DataFrame(story_lu_data)
        st.session_state.story_lu_df = st.data_editor(st.session_state.story_lu_df, use_container_width=True, hide_index=True, key='lu_editor')

# --- Main App Logic ---
steel_positions = generate_steel_positions(b_in, h_in, nb, nh, d_prime)
bar_area = np.pi * (bar_dia_mm / 10.0 / 2)**2
Ast_total = len(steel_positions) * bar_area
if bending_axis == 'Y (Weak Axis)':
    calc_b, calc_h, axis_label, M_col = h_in, b_in, "Y (Weak)", "M2"
    layers = get_layers_from_positions(steel_positions, 'Y')
else:
    calc_b, calc_h, axis_label, M_col = b_in, h_in, "X (Strong)", "M3"
    layers = get_layers_from_positions(steel_positions, 'X')
col1, col2 = st.columns([0.8, 1.2])
with col1:
    st.header("หน้าตัดเสา"); st.plotly_chart(draw_column_section_plotly(b_in, h_in, steel_positions, bar_dia_mm), use_container_width=True)
    st.markdown("---"); st.subheader("สรุปข้อมูล")
    st.metric("จำนวนเหล็กเสริมทั้งหมด", f"{len(steel_positions)} เส้น")
    st.metric("พื้นที่เหล็ก (Ast)", f"{Ast_total:.2f} ตร.ซม.")
    st.metric("อัตราส่วนเหล็กเสริม (ρg)", f"{Ast_total / (b_in*h_in):.2%}")
with col2:
    st.header(f"Interaction Diagram (แกน {axis_label})")
    
    # <<<---!!! จุดที่แก้ไข: นำเนื้อหาสูตรทั้งหมดกลับมาใส่ให้ครบถ้วน !!!--->>>
    with st.expander("แสดง/ซ่อนสูตรการคำนวณ"):
        st.markdown(r"""
        #### 1. Effective Flexural Stiffness ($EI_{eff}$)
        $$ EI_{eff} = \frac{0.4 \cdot E_c \cdot I_g}{1 + \beta_d} $$
        #### 2. Euler's Buckling Load ($P_c$)
        $$ P_c = \frac{\pi^2 \cdot EI_{eff}}{(k \cdot L_u)^2} $$
        #### 3. Equivalent Moment Factor ($C_m$)
        $$ C_m = 0.6 + 0.4 \frac{M_1}{M_2} \quad (0.4 \le C_m \le 1.0) $$
        #### 4. Moment Magnifier ($\delta_{ns}$)
        $$ \delta_{ns} = \frac{C_m}{1 - \frac{P_u}{0.75 \cdot P_c}} \geq 1.0 $$
        #### 5. Magnified Moment ($M_c$)
        $$ M_c = \delta_{ns} \cdot M_u $$
        #### 6. Minimum Moment ($M_{min}$)
        $$ M_{min} = P_u \cdot (1.5 + 0.03h) $$
        """)

    Pn_nom, Mn_nom, Pn_design, Mn_design, phi_Pn_max = calculate_interaction_diagram(fc, fy, calc_b, calc_h, layers, bar_area, 'Tied' if 'Tied' in column_type else 'Spiral')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Mn_nom, y=Pn_nom, mode='lines', name='Nominal Strength', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=Mn_design, y=np.minimum(Pn_design, phi_Pn_max), mode='lines', name='Design Strength', line=dict(color='red', width=3)))

    if df_loads is not None and 'selected_stories' in st.session_state and st.session_state.selected_stories:
        mask = (df_loads['Column'].isin(st.session_state.selected_columns)) & (df_loads['Story'].isin(st.session_state.selected_stories))
        column_data = df_loads[mask].copy()
        
        if not column_data.empty:
            column_data['P_ton'] = -column_data['P']
            column_data['Mu_ton_m'] = abs(column_data[M_col])
            column_data['Mc_ton_m'] = column_data['Mu_ton_m']
            if check_slenderness and st.session_state.story_lu_df is not None:
                lu_df = st.session_state.story_lu_df.rename(columns={'Lu (m)': 'Lu_m'})
                column_data['Story'] = column_data['Story'].astype(str)
                lu_df['Story'] = lu_df['Story'].astype(str)
                column_data = pd.merge(column_data, lu_df, on='Story', how='left')
                column_data['Pc_ton'] = column_data.apply(lambda row: calculate_euler_load(fc, calc_b, calc_h, beta_d, k_factor, row['Lu_m']), axis=1)
                grouping_keys = ['Story', 'Column', 'Unique Name', 'Output Case']
                if auto_calculate_cm:
                    cm_series = column_data.groupby(grouping_keys).apply(calculate_cm_for_group, M_col).rename('Cm')
                    column_data = pd.merge(column_data, cm_series, on=grouping_keys, how='left')
                else:
                    column_data['Cm'] = Cm_factor_manual
                results = column_data.apply(lambda row: get_magnified_moment_and_delta(row['P_ton'], row['Mu_ton_m'], row['Pc_ton'], row['Cm']), axis=1)
                column_data[['Mc_ton_m', 'delta_ns']] = pd.DataFrame(results.tolist(), index=column_data.index)
            
            column_data['M_design_ton_m'] = column_data['Mc_ton_m']
            if check_min_moment:
                column_data['M_min_ton_m'] = column_data.apply(lambda row: calculate_minimum_moment(row['P_ton'], calc_h), axis=1)
                column_data['M_design_ton_m'] = column_data[['Mc_ton_m', 'M_min_ton_m']].max(axis=1)

            hover_text_original = 'C:'+column_data['Column']+' S:'+column_data['Story']+' Sta:'+column_data['Station'].round(2).astype(str)+' Case:'+column_data['Output Case']
            fig.add_trace(go.Scatter(x=column_data['Mu_ton_m'], y=column_data['P_ton'], mode='markers', name='Original Loads (All Stations)', marker=dict(color='green', size=8, opacity=0.5), text=hover_text_original, hoverinfo='x+y+text'))
            if check_slenderness or check_min_moment:
                final_moment_col_name = 'M_design_ton_m'
                hover_text_final = 'C:'+column_data['Column']+' S:'+column_data['Story']+' Sta:'+column_data['Station'].round(2).astype(str)+' M_final='+column_data[final_moment_col_name].round(2).astype(str)
                fig.add_trace(go.Scatter(x=column_data[final_moment_col_name], y=column_data['P_ton'], mode='markers', name='Final Design Loads (All Stations)', marker=dict(color='purple', size=8, symbol='x', opacity=0.5), text=hover_text_final, hoverinfo='x+y+text'))
            
            idx = column_data.groupby(['Story', 'Column', 'Unique Name', 'Output Case'])['Station'].idxmax()
            summary_data = column_data.loc[idx]
            if check_slenderness and 'delta_ns' in summary_data.columns:
                failing_loads = summary_data[summary_data['delta_ns'] > 1.4]
                if not failing_loads.empty:
                    instability_failures = failing_loads[failing_loads['delta_ns'] >= 999]
                    st.warning(f"⚠️ คำเตือน: พบ {len(failing_loads)} รายการที่ Delta_ns > 1.4 (ที่ปลายบน)")
                    if not instability_failures.empty:
                        st.error(f"🚨 **พบ {len(instability_failures)} รายการที่เกิดการวิบัติจากการโก่งเดาะ (Pu ≥ 0.75Pc)**")
                    st.dataframe(failing_loads[['Story', 'Column', 'Output Case', 'delta_ns']].round(2))

    fig.update_layout(height=700, xaxis_title="Moment, M (Ton-m)", yaxis_title="Axial Load, P (Ton)", legend=dict(y=0.99, x=0.99))
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray'); fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray')
    st.plotly_chart(fig, use_container_width=True)

    if df_loads is not None and not column_data.empty:
        st.write("ข้อมูลแรงสำหรับเสาที่เลือก (แสดงทุก Station)")
        display_data = column_data.copy()
        display_cols = ['Story', 'Column', 'Unique Name', 'Station', 'Output Case', 'P', M_col, 'Mu_ton_m']
        if check_slenderness:
             display_cols.extend(['Cm', 'Pc_ton', 'delta_ns', 'Mc_ton_m'])
        if check_min_moment:
             if 'M_min_ton_m' not in display_cols: display_cols.append('M_min_ton_m')
             if 'M_design_ton_m' not in display_cols: display_cols.append('M_design_ton_m')
        
        final_display_cols = [col for col in display_cols if col in display_data.columns]
        st.dataframe(display_data[final_display_cols].sort_values(by=['Story', 'Column', 'Output Case', 'Station']).reset_index(drop=True).round(2))
