import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- ฟังก์ชันคำนวณต่างๆ ---

def generate_steel_positions(b, h, nb, nh, d_prime):
    """สร้างตำแหน่งของเหล็กเสริมในหน้าตัด"""
    bar_positions = []
    # เหล็กบน-ล่าง
    if nb > 0:
        x_coords_b = np.linspace(d_prime, b - d_prime, nb)
        for x in x_coords_b:
            bar_positions.append((x, d_prime))
            bar_positions.append((x, h - d_prime))
    # เหล็กด้านข้าง (ไม่รวมเหล็กมุมที่ใส่ไปแล้ว)
    if nh > 2:
        y_coords_h = np.linspace(d_prime, h - d_prime, nh)[1:-1]
        for y in y_coords_h:
            bar_positions.append((d_prime, y))
            bar_positions.append((b - d_prime, y))
    # ลบตำแหน่งที่ซ้ำกันออก (กรณี nb=2, nh=2) และเรียงลำดับ
    return sorted(list(set(bar_positions)))

def get_layers_from_positions(steel_positions, axis):
    """จัดกลุ่มเหล็กเสริมตามเลเยอร์สำหรับการคำนวณ"""
    layers = {}
    coord_index = 1 if axis == 'X' else 0 # 1=y-coord for X-axis bending, 0=x-coord for Y-axis
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
    d_t = max(steel_pos) # ระยะจากขอบรับแรงอัดถึงเหล็กเสริมแถวรับแรงดึงที่ไกลที่สุด
    
    Pn_nom_list, Mn_nom_list = [], []
    Pn_design_list, Mn_design_list = [], []
    
    # Pure Compression Point (P0)
    Pn_pc = 0.85 * fc * (Ag - Ast_total) + fy * Ast_total
    
    # คำนวณ phi และ ACI max axial load limit
    if column_type == 'Tied':
        phi_comp = 0.65
        alpha = 0.80
    else: # Spiral
        phi_comp = 0.75
        alpha = 0.85
    
    phi_Pn_max_aci = alpha * phi_comp * Pn_pc

    Pn_nom_list.append(Pn_pc)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pc * phi_comp)
    Mn_design_list.append(0.0)
    
    # Intermediate Points
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
            if fs > fy: fs = fy
            if fs < -fy: fs = -fy
            
            # ปรับปรุงการคำนวณแรงในเหล็กโดยหักพื้นที่คอนกรีตที่ถูกแทนที่
            force = fs * As_i
            if fs >= -0.85 * fc: # ถ้าเหล็กรับแรงอัด
                 force -= 0.85 * fc * As_i
            
            Pn_s += force
            Mn_s += force * (h / 2.0 - d_i)
            
        Pn = Cc + Pn_s
        Mn = Mc + Mn_s
        
        if Mn >= 0:
            epsilon_t = epsilon_c_max * (d_t - c) / c if c > 0 else float('inf')
            
            # Strength Reduction Factor (phi) variation
            if column_type == 'Tied':
                if epsilon_t <= epsilon_y: phi = 0.65
                elif epsilon_t >= 0.005: phi = 0.90
                else: phi = 0.65 + 0.25 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)
            else: # Spiral
                if epsilon_t <= epsilon_y: phi = 0.75
                elif epsilon_t >= 0.005: phi = 0.90
                else: phi = 0.75 + 0.15 * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)

            Pn_nom_list.append(Pn)
            Mn_nom_list.append(Mn)
            Pn_design_list.append(Pn * phi)
            Mn_design_list.append(Mn * phi)
            
    # Pure Tension Point
    Pn_pt = -fy * Ast_total
    phi_pt = 0.90
    Pn_nom_list.append(Pn_pt)
    Mn_nom_list.append(0.0)
    Pn_design_list.append(Pn_pt * phi_pt)
    Mn_design_list.append(0.0)
    
    Pn_nom, Mn_nom = np.array(Pn_nom_list), np.array(Mn_nom_list)
    Pn_design, Mn_design = np.array(Pn_design_list), np.array(Mn_design_list)
    
    # Sort values by Pn descending for clean plotting
    sort_indices = np.argsort(Pn_nom)[::-1]
    
    # Convert units to Ton and Ton-m
    return (Pn_nom[sort_indices]/1000, Mn_nom[sort_indices]/100000,
            Pn_design[sort_indices]/1000, Mn_design[sort_indices]/100000,
            phi_Pn_max_aci / 1000)

# --- ฟังก์ชันสำหรับความชะลูด (Slenderness) ---
def calculate_euler_load(fc, b, h, beta_d, k, L_unsupported_m):
    """
    คำนวณหา Euler's buckling load (Pc) สำหรับ Non-sway frame ตาม ACI 318
    """
    Ec = 15100 * np.sqrt(fc)  # Modulus of Elasticity ของคอนกรีต (ksc)
    Ig = (b * h**3) / 12     # Gross Moment of Inertia (cm^4)
    EI_eff = (0.4 * Ec * Ig) / (1 + beta_d)
    
    Lu_cm = L_unsupported_m * 100
    if (k * Lu_cm) == 0: return float('inf')

    Pc_kg = (np.pi**2 * EI_eff) / (k * Lu_cm)**2
    Pc_ton = Pc_kg / 1000 # แปลงเป็นหน่วย ตัน
    return Pc_ton

def get_magnified_moment(Pu_ton, Mu_ton_m, Pc_ton, Cm):
    """คำนวณโมเมนต์ที่ขยายค่าแล้ว (Magnified Moment)"""
    if Pu_ton <= 0 or Pc_ton <= 0: # ถ้าเป็นแรงดึงหรือไม่สามารถคำนวณ Pc ได้
        return Mu_ton_m

    # Pu ต้องเป็นค่าบวกในการคำนวณ
    Pu_abs = abs(Pu_ton)
    
    denominator = (1 - (Pu_abs / (0.75 * Pc_ton)))
    if denominator <= 0: # ป้องกันการหารด้วยศูนย์หรือค่าลบ
        return float('inf') # หรือจัดการเป็นกรณีพิเศษ

    delta_ns = Cm / denominator
    delta_ns = max(1.0, delta_ns)
    
    Mc = delta_ns * Mu_ton_m
    return Mc


# --- ฟังก์ชันวาดหน้าตัดเสาด้วย Plotly ---

def draw_column_section_plotly(b, h, steel_positions, bar_dia_mm):
    """วาดรูปหน้าตัดเสาด้วย Plotly"""
    fig = go.Figure()
    
    # Concrete section
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h,
                  line=dict(color="Black", width=2), fillcolor="LightGrey",
                  layer='below')
                  
    # Steel bars
    bar_dia_cm = bar_dia_mm / 10.0
    bar_x = [pos[0] for pos in steel_positions]
    bar_y = [pos[1] for pos in steel_positions]
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers',
        marker=dict(color='DarkSlateGray', size=bar_dia_cm * 5, symbol='circle',
                    line=dict(color='Black', width=1)),
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title="Column Cross-Section",
        xaxis_title="Width, b (cm)",
        yaxis_title="Height, h (cm)",
        yaxis_scaleanchor="x",
        xaxis_range=[-b*0.1, b*1.1],
        yaxis_range=[-h*0.1, h*1.1],
        width=500, height=500,
        showlegend=False
    )
    return fig

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🏗️ Column Interaction Diagram Generator (ACI Compliant)")
st.write("เครื่องมือสำหรับสร้าง Interaction Diagram และตรวจสอบผลของความชะลูด (Slenderness Effects)")

# --- Sidebar Inputs ---

with st.sidebar:
    st.header("ใส่ข้อมูลหน้าตัดเสา")
    
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

    # --- ส่วนที่เพิ่มเข้ามาสำหรับ Slenderness ---
    st.markdown("---")
    with st.expander("ข้อมูลความชะลูด (Slenderness)", expanded=False):
        check_slenderness = st.checkbox("พิจารณาผลของความชะลูด")
        L_unsupported = st.number_input("ความยาวเสาที่ไม่มีการค้ำยัน, Lu (เมตร)", min_value=0.1, value=3.0, disabled=not check_slenderness)
        k_factor = st.number_input("ตัวคูณความยาวประสิทธิผล, k", min_value=0.5, value=1.0, help="1.0 สำหรับ Pinned-Pinned, 0.65 สำหรับ Fixed-Fixed", disabled=not check_slenderness)
        Cm_factor = st.number_input("Cm Factor", min_value=0.4, max_value=1.0, value=1.0, help="โดยทั่วไปใช้ 1.0 สำหรับโครงข้อแข็งที่ไม่มีการเซ (Non-sway)", disabled=not check_slenderness)
        beta_d = st.number_input("อัตราส่วนแรงกระทำคงที่ต่อแรงกระทำทั้งหมด, βd", min_value=0.0, max_value=1.0, value=0.6, help="ใช้สำหรับผลการสั่นไหว (Creep effects)", disabled=not check_slenderness)
        
    st.markdown("---")
    st.header("ตรวจสอบแรงจากไฟล์ CSV")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ load_export.csv", type=["csv"])

    df_loads = None
    if uploaded_file is not None:
        try:
            df_loads = pd.read_csv(uploaded_file)
            required_cols = {'Story', 'Column', 'P', 'M2', 'M3', 'Output Case'}
            if not required_cols.issubset(df_loads.columns):
                st.sidebar.error(f"ไฟล์ CSV ต้องมีคอลัมน์: {', '.join(required_cols)}")
                df_loads = None
            else:
                column_options = sorted(df_loads['Column'].unique())
                selected_columns = st.multiselect("เลือกหมายเลขเสาที่ต้องการตรวจสอบ:", column_options)
                if selected_columns:
                    filtered_df_for_stories = df_loads[df_loads['Column'].isin(selected_columns)]
                    story_options = sorted(filtered_df_for_stories['Story'].unique())
                    selected_stories = st.multiselect("เลือกชั้นที่ต้องการตรวจสอบ:", story_options)

        except Exception as e:
            st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

# --- Main App Logic ---

steel_positions = generate_steel_positions(b_in, h_in, nb, nh, d_prime)
bar_area = np.pi * (bar_dia_mm / 10.0 / 2)**2
total_bars = len(steel_positions)
Ast_total = total_bars * bar_area
Ag_total = b_in * h_in
rho_g = (Ast_total / Ag_total)

if bending_axis == 'Y (Weak Axis)':
    calc_b, calc_h = h_in, b_in
    axis_label = "Y (Weak)"
    layers = get_layers_from_positions(steel_positions, 'Y')
else:
    calc_b, calc_h = b_in, h_in
    axis_label = "X (Strong)"
    layers = get_layers_from_positions(steel_positions, 'X')

col1, col2 = st.columns([0.8, 1.2])

with col1:
    st.header("หน้าตัดเสา (Visualization)")
    fig_section = draw_column_section_plotly(b_in, h_in, steel_positions, bar_dia_mm)
    st.plotly_chart(fig_section, use_container_width=True)

    st.markdown("---")
    st.subheader("สรุปข้อมูลเหล็กเสริม")
    st.metric(label="จำนวนเหล็กเสริมทั้งหมด", value=f"{total_bars} เส้น")
    st.metric(label="พื้นที่หน้าตัดเหล็กทั้งหมด (Ast)", value=f"{Ast_total:.2f} ตร.ซม.")
    st.metric(label="อัตราส่วนเหล็กเสริม (ρg)", value=f"{rho_g:.2%}")

with col2:
    st.header(f"Interaction Diagram (แกน {axis_label})")
    
    with st.spinner("กำลังคำนวณ..."):
        col_type_val = 'Tied' if column_type == 'เหล็กปลอกเดี่ยว (Tied)' else 'Spiral'
        
        Pn_nom, Mn_nom, Pn_design, Mn_design, phi_Pn_max = calculate_interaction_diagram(
            fc, fy, calc_b, calc_h, layers, bar_area, column_type=col_type_val
        )
        
        if Pn_nom is not None:
            fig_diagram = go.Figure()

            # Plot Nominal Strength
            fig_diagram.add_trace(go.Scatter(x=Mn_nom, y=Pn_nom, mode='lines', name='Nominal Strength (Pn, Mn)', line=dict(color='blue', dash='dash')))
            
            # Plot ACI-Compliant Design Strength (capped)
            fig_diagram.add_trace(go.Scatter(
                x=Mn_design, 
                y=np.minimum(Pn_design, phi_Pn_max),
                mode='lines', 
                name='Design Strength (ΦPn, ΦMn)',
                line=dict(color='red', width=3),
                fill='tozeroy' # заливка области под кривой
            ))

            # --- ส่วนจัดการการพล็อตแรงจาก CSV ---
            if df_loads is not None and selected_columns and selected_stories:
                mask = (df_loads['Column'].isin(selected_columns)) & (df_loads['Story'].isin(selected_stories))
                column_data = df_loads[mask].copy()
                
                if not column_data.empty:
                    column_data['P_ton'] = -column_data['P']
                    M_col = 'M3' if bending_axis.startswith('X') else 'M2'
                    column_data['Mu_ton_m'] = abs(column_data[M_col])

                    # Plot Original Loads
                    fig_diagram.add_trace(go.Scatter(
                        x=column_data['Mu_ton_m'], y=column_data['P_ton'], mode='markers',
                        name='Original Loads (Pu, Mu)',
                        marker=dict(color='green', size=8, symbol='circle'),
                        text='C:' + column_data['Column'] + ' S:' + column_data['Story'].astype(str) + ' Case:' + column_data['Output Case'],
                        hoverinfo='x+y+text'
                    ))

                    # If slenderness is checked, calculate and plot magnified loads
                    if check_slenderness:
                        Pc_ton = calculate_euler_load(fc, calc_b, calc_h, beta_d, k_factor, L_unsupported)
                        st.sidebar.info(f"คำนวณ Euler Load (Pc) = {Pc_ton:.2f} ตัน")

                        column_data['Mc_ton_m'] = column_data.apply(
                            lambda row: get_magnified_moment(row['P_ton'], row['Mu_ton_m'], Pc_ton, Cm_factor),
                            axis=1
                        )

                        fig_diagram.add_trace(go.Scatter(
                            x=column_data['Mc_ton_m'], y=column_data['P_ton'], mode='markers',
                            name='Magnified Loads (Pu, Mc)',
                            marker=dict(color='purple', size=10, symbol='x'),
                            text='C:' + column_data['Column'] + ' S:' + column_data['Story'].astype(str) + ' Mc=' + column_data['Mc_ton_m'].round(2).astype(str),
                            hoverinfo='x+y+text'
                        ))

            # Finalize Diagram Layout
            fig_diagram.update_layout(
                title=f"P-M Interaction Diagram ({axis_label} Axis)",
                xaxis_title="Moment, M (Ton-m)",
                yaxis_title="Axial Load, P (Ton)",
                legend_title="Legend",
                height=700
            )
            fig_diagram.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray', range=[0, max(Mn_design.max(), 1)*1.1])
            fig_diagram.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightGray')

            st.plotly_chart(fig_diagram, use_container_width=True)

            # Display Loads DataFrame
            if df_loads is not None and selected_columns and selected_stories:
                st.write(f"ข้อมูลแรงสำหรับเสา **{', '.join(selected_columns)}** ที่ชั้น **{', '.join(map(str, selected_stories))}**")
                display_cols = ['Story', 'Column', 'Output Case', 'P', 'M2', 'M3']
                if check_slenderness and 'Mc_ton_m' in column_data.columns:
                     # Create a temporary column for display, matching the original M column name
                    M_col_magnified = 'M3_magnified' if bending_axis.startswith('X') else 'M2_magnified'
                    column_data[M_col_magnified] = column_data['Mc_ton_m']
                    display_cols.append(M_col_magnified)

                st.dataframe(column_data[display_cols].reset_index(drop=True), use_container_width=True)
        else:
            st.error("เกิดข้อผิดพลาดในการคำนวณ")

st.markdown("---")
