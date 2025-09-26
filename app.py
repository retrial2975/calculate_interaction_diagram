import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ฟังก์ชันคำนวณต่างๆ ---

def generate_steel_positions(b, h, nb, nh, d_prime):
    """สร้างตำแหน่งของเหล็กเสริมในหน้าตัด"""
    bar_positions = []
    if nb > 0:
        x_coords_b = np.linspace(d_prime, b - d_prime, nb)
        for x in x_coords_b:
            bar_positions.append((x, d_prime))
            bar_positions.append((x, h - d_prime))
    if nh > 2:
        y_coords_h = np.linspace(d_prime, h - d_prime, nh)[1:-1]
        for y in y_coords_h:
            bar_positions.append((d_prime, y))
            bar_positions.append((b - d_prime, y))
    return sorted(list(set(bar_positions)))

def get_layers_from_positions(steel_positions, axis):
    """จัดกลุ่มเหล็กเสริมตามเลเยอร์สำหรับการคำนวณ"""
    layers = {}
    coord_index = 1 if axis == 'X' else 0
    for pos in steel_positions:
        layer_pos = pos[coord_index]
        if layer_pos in layers:
            layers[layer_pos] += 1
        else:
            layers[layer_pos] = 1
    return layers

def calculate_interaction_diagram(fc, fy, b, h, layers, bar_area, column_type='Tied'):
    """คำนวณหาค่า Pn, Mn สำหรับ Interaction Diagram พร้อมการจำกัดค่าตาม ACI"""
    Es = 2.0e6
    epsilon_c_max = 0.003
    epsilon_y = fy / Es
    if fc <= 280: beta1 = 0.85
    elif fc < 560: beta1 = 0.85 - 0.05 * (fc - 280) / 70
    else: beta1 = 0.65
    
    steel_pos = sorted(layers.keys())
    steel_areas = [layers[pos] * bar_area for pos in steel_pos]
    Ast_total = sum(steel_areas)
    Ag = b * h
    d_t = max(steel_pos)
    
    Pn_nom_list, Mn_nom_list = [], []
    Pn_design_list, Mn_design_list = [], []
    
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    
    if column_type == 'Tied':
        phi_comp, alpha = 0.65, 0.80
    else: # Spiral
        phi_comp, alpha = 0.75, 0.85
    
    phi_Pn_max_aci = alpha * phi_comp * Pn_pc

    Pn_nom_list.append(Pn_pc)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pc * phi_comp)
    Mn_design_list.append(0.0)
    
    c_values = np.logspace(np.log10(0.01), np.log10(h * 5), 300)
    for c in c_values:
        a = beta1 * c
        if a > h: a = h
        
        Cc = 0.85 * fc * a * b
        Mc = Cc * (h / 2.0 - a / 2.0)
        
        Pn_s, Mn_s = 0.0, 0.0
        for i, d_i in enumerate(steel_pos):
            As_i = steel_areas[i]
            epsilon_s = epsilon_c_max * (c - d_i) / c
            fs = Es * epsilon_s
            fs = np.clip(fs, -fy, fy) # Simpler way to cap fs
            
            force = fs * As_i - (0.85 * fc * As_i if fs >= -0.85 * fc else 0)
            
            Pn_s += force
            Mn_s += force * (h / 2.0 - d_i)
            
        Pn = Cc + Pn_s
        Mn = Mc + Mn_s
        
        if Mn >= 0:
            epsilon_t = epsilon_c_max * (d_t - c) / c if c > 0 else float('inf')
            
            if column_type == 'Tied':
                phi = np.interp(epsilon_t, [epsilon_y, 0.005], [0.65, 0.90])
            else: # Spiral
                phi = np.interp(epsilon_t, [epsilon_y, 0.005], [0.75, 0.90])
            phi = np.clip(phi, min(phi_comp, 0.75), 0.90)

            Pn_nom_list.append(Pn)
            Mn_nom_list.append(Mn)
            Pn_design_list.append(Pn * phi)
            Mn_design_list.append(Mn * phi)
            
    Pn_pt = -fy * Ast_total
    Pn_nom_list.append(Pn_pt)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pt * 0.90)
    Mn_design_list.append(0.0)
    
    Pn_nom, Mn_nom = np.array(Pn_nom_list), np.array(Mn_nom_list)
    Pn_design, Mn_design = np.array(Pn_design_list), np.array(Mn_design_list)
    
    sort_indices = np.argsort(Pn_nom)[::-1]
    
    return (Pn_nom[sort_indices]/1000, Mn_nom[sort_indices]/100000,
            Pn_design[sort_indices]/1000, Mn_design[sort_indices]/100000,
            phi_Pn_max_aci / 1000)

# --- ฟังก์ชันสำหรับความชะลูด (Slenderness) ---
def calculate_euler_load(fc, b, h, beta_d, k, L_unsupported_m):
    Ec = 15100 * np.sqrt(fc)
    Ig = (b * h**3) / 12
    EI_eff = (0.4 * Ec * Ig) / (1 + beta_d)
    
    Lu_cm = L_unsupported_m * 100
    if (k * Lu_cm) == 0: return float('inf')

    Pc_kg = (np.pi**2 * EI_eff) / (k * Lu_cm)**2
    return Pc_kg / 1000

# <<<---!!! จุดที่แก้ไข 1: ให้ฟังก์ชันคืนค่า delta_ns ด้วย !!!--->>>
def get_magnified_moment_and_delta(Pu_ton, Mu_ton_m, Pc_ton, Cm):
    """คำนวณโมเมนต์ที่ขยายค่าแล้ว (Mc) และคืนค่า delta_ns"""
    if Pu_ton <= 0 or Pc_ton <= 0:
        return Mu_ton_m, 1.0 # คืนค่า delta_ns เป็น 1.0

    Pu_abs = abs(Pu_ton)
    denominator = (1 - (Pu_abs / (0.75 * Pc_ton)))
    if denominator <= 0:
        return float('inf'), float('inf')

    delta_ns = max(1.0, Cm / denominator)
    Mc = delta_ns * Mu_ton_m
    return Mc, delta_ns # คืน 2 ค่า

# --- ฟังก์ชันวาดหน้าตัดเสา ---
def draw_column_section_plotly(b, h, steel_positions, bar_dia_mm):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h,
                  line=dict(color="Black", width=2), fillcolor="LightGrey", layer='below')
    
    bar_dia_cm = bar_dia_mm / 10.0
    bar_x = [pos[0] for pos in steel_positions]
    bar_y = [pos[1] for pos in steel_positions]
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers',
        marker=dict(color='DarkSlateGray', size=bar_dia_cm * 5, line=dict(color='Black', width=1)),
        hoverinfo='none'))
    
    fig.update_layout(title="Column Cross-Section", xaxis_title="Width, b (cm)", yaxis_title="Height, h (cm)",
                      yaxis_scaleanchor="x", xaxis_range=[-b*0.1, b*1.1], yaxis_range=[-h*0.1, h*1.1],
                      width=500, height=500, showlegend=False)
    return fig

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator (ACI Compliant)")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram และตรวจสอบผลของความชะลูด (Slenderness Effects)")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    # ... (ส่วนนี้เหมือนเดิมทั้งหมด) ...
    column_type = st.radio("ประเภทของเหล็กปลอก:", ('เหล็กปลอกเดี่ยว (Tied)', 'เหล็กปลอกเกลียว (Spiral)'))
    bending_axis = st.radio("เลือกแกนที่ต้องการคำนวณโมเมนต์:", ('X (Strong Axis)', 'Y (Weak Axis)'))
    
    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("fc' (ksc)", min_value=1.0, value=280.0)
        fy = st.number_input("fy (ksc)", min_value=1.0, value=4000.0)
        
    with st.expander("ขนาดหน้าตัด", expanded=True):
        b_in = st.number_input("ความกว้างหน้าตัด, b (cm)", min_value=1.0, value=40.0)
        h_in = st.number_input("ความลึกหน้าตัด, h (cm)", min_value=1.0, value=60.0)
        
    with st.expander("ข้อมูลเหล็กเสริม", expanded=True):
        d_prime = st.number_input("ระยะขอบถึงศูนย์กลางเหล็ก, d' (cm)", min_value=1.0, value=6.0)
        bar_dia_mm = st.selectbox("ขนาดเหล็กเสริม", [12, 16, 20, 25, 28, 32], index=3)
        st.markdown("**การจัดเรียงเหล็ก (รวมเหล็กมุม)**")
        nb = st.number_input("จำนวนเหล็กในด้านขนานแกน b (บน-ล่าง)", min_value=2, value=5)
        nh = st.number_input("จำนวนเหล็กในด้านขนานแกน h (ข้าง)", min_value=2, value=3)

    st.markdown("---")
    with st.expander("ข้อมูลความชะลูด (Slenderness)", expanded=False):
        check_slenderness = st.checkbox("พิจารณาผลของความชะลูด")
        k_factor = st.number_input("k-factor", value=1.0, disabled=not check_slenderness)
        Cm_factor = st.number_input("Cm Factor", value=1.0, disabled=not check_slenderness)
        beta_d = st.number_input("βd", value=0.6, disabled=not check_slenderness)
        
    st.markdown("---")
    st.header("ตรวจสอบแรงจากไฟล์ CSV")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ load_export.csv", type=["csv"])

    # ... (ส่วนจัดการไฟล์และเลือกชั้น เหมือนเดิมทั้งหมด) ...
    if 'selected_columns' not in st.session_state: st.session_state.selected_columns = []
    if 'selected_stories' not in st.session_state: st.session_state.selected_stories = []
    df_loads, story_lu_editor = None, None
    if uploaded_file is not None:
        try:
            df_loads = pd.read_csv(uploaded_file)
            required_cols = {'Story', 'Column', 'P', 'M2', 'M3', 'Output Case'}
            if not required_cols.issubset(df_loads.columns):
                st.sidebar.error(f"ไฟล์ CSV ต้องมีคอลัมน์: {', '.join(required_cols)}")
                df_loads = None
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

                    if selected_stories and check_slenderness:
                        st.markdown("**กรอกความสูงของแต่ละชั้น (Lu):**")
                        story_lu_df = pd.DataFrame({'Story': sorted(selected_stories), 'Lu (m)': [3.0] * len(selected_stories)})
                        story_lu_editor = st.data_editor(story_lu_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

# --- Main App Logic ---
# ... (ส่วนเตรียมข้อมูลเหมือนเดิม) ...
steel_positions = generate_steel_positions(b_in, h_in, nb, nh, d_prime)
bar_area = np.pi * (bar_dia_mm / 10.0 / 2)**2
Ast_total = len(steel_positions) * bar_area
rho_g = Ast_total / (b_in * h_in)

if bending_axis == 'Y (Weak Axis)':
    calc_b, calc_h = h_in, b_in
    axis_label, M_col_name = "Y (Weak)", "M2"
    layers = get_layers_from_positions(steel_positions, 'Y')
else:
    calc_b, calc_h = b_in, h_in
    axis_label, M_col_name = "X (Strong)", "M3"
    layers = get_layers_from_positions(steel_positions, 'X')

col1, col2 = st.columns([0.8, 1.2])
with col1:
    # ... (ส่วนแสดงหน้าตัดและข้อมูลเหล็กเสริม เหมือนเดิม) ...
    st.header("หน้าตัดเสา"); st.plotly_chart(draw_column_section_plotly(b_in, h_in, steel_positions, bar_dia_mm), use_container_width=True)
    st.markdown("---"); st.subheader("สรุปข้อมูล")
    st.metric("จำนวนเหล็กเสริมทั้งหมด", f"{len(steel_positions)} เส้น")
    st.metric("พื้นที่หน้าตัดเหล็ก (Ast)", f"{Ast_total:.2f} ตร.ซม.")
    st.metric("อัตราส่วนเหล็กเสริม (ρg)", f"{rho_g:.2%}")

with col2:
    st.header(f"Interaction Diagram (แกน {axis_label})")
    # ... (ส่วนแสดงสูตรเหมือนเดิม) ...
    with st.expander("แสดง/ซ่อนสูตรการคำนวณความชะลูด (ACI Moment Magnifier)"):
        st.markdown(r"""...สูตรต่างๆ...""")
    
    Pn_nom, Mn_nom, Pn_design, Mn_design, phi_Pn_max = calculate_interaction_diagram(fc, fy, calc_b, calc_h, layers, bar_area, 'Tied' if 'Tied' in column_type else 'Spiral')
    
    fig_diagram = go.Figure()
    fig_diagram.add_trace(go.Scatter(x=Mn_nom, y=Pn_nom, mode='lines', name='Nominal Strength', line=dict(color='blue', dash='dash')))
    fig_diagram.add_trace(go.Scatter(x=Mn_design, y=np.minimum(Pn_design, phi_Pn_max), mode='lines', name='Design Strength (ΦPn, ΦMn)', line=dict(color='red', width=3)))

    if df_loads is not None and st.session_state.selected_columns and st.session_state.selected_stories:
        mask = (df_loads['Column'].isin(st.session_state.selected_columns)) & (df_loads['Story'].isin(st.session_state.selected_stories))
        column_data = df_loads[mask].copy()
        
        if not column_data.empty:
            column_data['P_ton'] = -column_data['P']
            column_data['Mu_ton_m'] = abs(column_data[M_col_name])

            fig_diagram.add_trace(go.Scatter(x=column_data['Mu_ton_m'], y=column_data['P_ton'], mode='markers', name='Original Loads (Pu, Mu)', marker=dict(color='green', size=8), text='C:'+column_data['Column']+' S:'+column_data['Story'].astype(str)+' Case:'+column_data['Output Case'], hoverinfo='x+y+text'))

            if check_slenderness and story_lu_editor is not None:
                story_lu_map = pd.Series(story_lu_editor['Lu (m)'].values, index=story_lu_editor['Story']).to_dict()
                story_pc_map = {story: calculate_euler_load(fc, calc_b, calc_h, beta_d, k_factor, lu) for story, lu in story_lu_map.items()}
                st.sidebar.markdown("**Euler Load (Pc) ที่คำนวณได้:**"); st.sidebar.json({k: f"{v:.2f} ตัน" for k, v in story_pc_map.items()})

                column_data['Pc_ton'] = column_data['Story'].map(story_pc_map)
                
                # <<<---!!! จุดที่แก้ไข 2: รับค่า Mc และ delta_ns มาใส่ในคอลัมน์ใหม่ !!!--->>>
                results = column_data.apply(lambda row: get_magnified_moment_and_delta(row['P_ton'], row['Mu_ton_m'], row['Pc_ton'], Cm_factor), axis=1)
                column_data[['Mc_ton_m', 'delta_ns']] = pd.DataFrame(results.tolist(), index=column_data.index)

                fig_diagram.add_trace(go.Scatter(x=column_data['Mc_ton_m'], y=column_data['P_ton'], mode='markers', name='Magnified Loads (Pu, Mc)', marker=dict(color='purple', size=10, symbol='x'), text='C:'+column_data['Column']+' S:'+column_data['Story'].astype(str)+' δns='+column_data['delta_ns'].round(2).astype(str), hoverinfo='x+y+text'))
    
    # ... (ส่วน Layout กราฟเหมือนเดิม) ...
    fig_diagram.update_layout(height=700, xaxis_title="Moment, M (Ton-m)", yaxis_title="Axial Load, P (Ton)", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig_diagram.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray')
    fig_diagram.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray')
    st.plotly_chart(fig_diagram, use_container_width=True)

    # <<<---!!! จุดที่แก้ไข 3: เพิ่มการตรวจสอบและแสดง Warning !!!--->>>
    if check_slenderness and 'delta_ns' in column_data.columns:
        failing_loads = column_data[column_data['delta_ns'] > 1.4]
        if not failing_loads.empty:
            st.warning(f"⚠️ คำเตือน: พบ {len(failing_loads)} รายการที่ค่า Delta_ns > 1.4 (ACI 318-14 Sec. 6.6.4.5.1)")
            st.write("รายการเหล่านี้มีความเสี่ยงสูงต่อการโก่งเดาะ แนะนำให้เพิ่มขนาดหน้าตัดเสา:")
            st.dataframe(failing_loads[['Story', 'Column', 'Output Case', 'P_ton', 'Mu_ton_m', 'delta_ns']].round(2), use_container_width=True)

    if df_loads is not None and st.session_state.selected_columns and st.session_state.selected_stories:
        st.write(f"ข้อมูลแรงสำหรับเสาที่เลือก")
        display_cols = ['Story', 'Column', 'Output Case', 'P', M_col_name]
        if check_slenderness and 'Mc_ton_m' in column_data.columns:
            column_data[f'{M_col_name}_magnified'] = column_data['Mc_ton_m']
            display_cols.extend([f'{M_col_name}_magnified', 'Pc_ton', 'delta_ns'])
        st.dataframe(column_data[display_cols].reset_index(drop=True).round(2), use_container_width=True)
