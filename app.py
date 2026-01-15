import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI ทำนายผลใบหยุด", page_icon="🤖")
st.title("🤖 AI ผู้ช่วยทำนายผลใบหยุด")

# --- 2. ฟังก์ชันโหลดข้อมูล ---
@st.cache_data
def load_data():
    folder_path = '.' 
    all_dfs = []
    
    if not os.path.exists(folder_path):
        return None

    files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.csv'))]
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith('.csv'):
                df_temp = pd.read_csv(file_path)
            else:
                df_temp = pd.read_excel(file_path)
            
            # เคลียร์ชื่อหัวตาราง
            df_temp.columns = df_temp.columns.str.strip()
            
            # จัดการชื่อคอลัมน์ให้เป็นมาตรฐาน
            rename_map = {
                'Full Description': 'Full_Description',
                'Problem': 'Full_Description',
                'Defect': 'Full_Description',
                'Decision Result': 'Decision result',
                'Result': 'Decision result',
                'Status': 'Decision result',
                'Cable Type': 'Cables type',  # เผื่อบางไฟล์ชื่อไม่ตรง
                'Wire Type': 'Cables type'
            }
            df_temp = df_temp.rename(columns=rename_map)
            
            # เช็คว่ามีคอลัมน์สำคัญครบไหม
            if 'Full_Description' in df_temp.columns:
                all_dfs.append(df_temp)
                
        except Exception as e:
            continue

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        
        # คอลัมน์ที่ต้องการแสดงผล
        cols_display = ['No of HoldTag', 'Customer', 'Cables type', 'Decision result', 'Date', 'Recheck', 'Full_Description']
        
        # จัดการค่าว่าง (NaN)
        for col in cols_display:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '-')
        
        # ตรวจสอบคอลัมน์ค้นหา (ถ้าไม่มีให้สร้างไว้กัน Error)
        if 'Full_Description' not in df.columns: df['Full_Description'] = "-"
        if 'Cables type' not in df.columns: df['Cables type'] = "-"
        if 'Customer' not in df.columns: df['Customer'] = "-"
            
        return df
    
    return None

df = load_data()

# --- 3. ส่วนเตรียมสมอง AI ---
if df is not None:
    search_col = 'Full_Description'
    result_col = 'Decision result'

    # 🔥 [ส่วนที่ 1] คำศัพท์เหมือน (Synonyms)
    synonyms = {
        "T/S": "Tensile Strength แรงดึง",
        "B/S": "Breaking Strength แรงดึงขาด",
        "MG": "Master Batch เม็ดสี",
        "OD": "Outer Diameter ขนาดภายนอก",
        "ID": "Inner Diameter ขนาดภายใน",
        "Elong": "Elongation ยืด",
        "AV": "Automotive Wire สายรถยนต์", # ตัวอย่างเพิ่มชนิดสาย
    }

    def enrich_text(text):
        text = str(text).lower()
        for key, value in synonyms.items():
            if key.lower() in text: text += " " + value
            if value.lower() in text: text += " " + key
        return text

    # 🔥 [ส่วนที่ 2] รวมร่างคอลัมน์เพื่อค้นหา (Combined Search) 🔥
    # เอา ชนิดสาย + ลูกค้า + ปัญหา มารวมกัน เพื่อให้ค้นหาได้ทุกอย่าง
    df['Combined_Search'] = df['Cables type'].astype(str) + " " + \
                            df['Customer'].astype(str) + " " + \
                            df['Full_Description'].astype(str)
    
    # ใส่คำศัพท์เพิ่มเข้าไปในข้อมูลรวม
    df['AI_Input'] = df['Combined_Search'].apply(enrich_text)

    # สร้าง AI (Vectorizer)
    # ngram_range=(2,5) ช่วยให้หาคำสั้นๆ หรือรหัสสายไฟได้แม่นขึ้น
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(df['AI_Input'])

    st.success(f"✅ ฐานข้อมูลพร้อมใช้งาน ({len(df)} รายการ)")

    # --- 4. ส่วนหน้าจอค้นหา ---
    top_n = st.slider("จำนวนเคสที่แสดง", 1, 10, 3)
    
    # เปลี่ยนคำอธิบายให้ User รู้ว่าค้นได้หลายอย่าง
    query_input = st.text_input("ค้นหา (ระบุชนิดสาย, ปัญหา, หรือลูกค้า):", 
                                placeholder="เช่น AVVSS, สายเป็นคลื่น, T/S ต่ำ...")

    if query_input:
        query_enriched = enrich_text(query_input)
        
        query_vec = vectorizer.transform([query_enriched])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-top_n:][::-1]

        st.markdown("---")
        
        # คำนวณ % ความน่าจะเป็น
        results_to_show = []
        for idx in top_indices:
            score = similarity[idx]
            if score > 0.01: 
                results_to_show.append((idx, score))
        
        total_raw_score = sum([s for _, s in results_to_show])
        
        if not results_to_show:
            st.info("ไม่พบข้อมูลที่ตรงกัน")
        else:
            for idx, original_score in results_to_show:
                row = df.iloc[idx]
                decision = str(row.get(result_col, '-'))
                
                # คำนวณ %
                if total_raw_score > 0:
                    normalized_percent = (original_score / total_raw_score) * 100
                else:
                    normalized_percent = 0
                
                # Logic สี
                decision_lower = decision.lower()
                status_icon = "🟡"
                msg_func = st.warning
                if any(x in decision_lower for x in ['scrap', 'reject', 'ทิ้ง', 'ไม่ผ่าน', 'เสีย']):
                    status_icon = "🔴"
                    msg_func = st.error
                elif any(x in decision_lower for x in ['pass', 'ผ่าน', 'accept', 'ok', 'อนุโลม']):
                    status_icon = "🟢"
                    msg_func = st.success
                
                # แสดงหัวข้อผลลัพธ์
                msg_func(f"{status_icon} ผล: {decision} ({normalized_percent:.0f}%)")

                # แสดงรายละเอียด (จัดเรียงใหม่ให้อ่านง่าย)
                with st.expander(f"รายละเอียดใบหยุด: {row.get('No of HoldTag', '-')}"):
                    st.markdown(f"**📌 ชนิดสาย:** `{row.get('Cables type', '-')}`")
                    st.markdown(f"**🏢 ลูกค้า:** {row.get('Customer', '-')}")
                    st.markdown(f"**⚠️ อาการเสีย:** {row.get(search_col, '-')}")
                    st.markdown(f"**📅 วันที่:** {row.get('Date', '-')}")
                    
                    if row.get('Recheck', '-') != '-':
                        st.info(f"**Recheck:** {row.get('Recheck', '-')}")

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล! กรุณาตรวจสอบว่ามีไฟล์ Excel (.xlsx) วางอยู่คู่กับไฟล์ app.py หรือไม่")
