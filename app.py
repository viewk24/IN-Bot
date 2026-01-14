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
            
            df_temp.columns = df_temp.columns.str.strip()
            
            rename_map = {
                'Full Description': 'Full_Description',
                'Problem': 'Full_Description',
                'Defect': 'Full_Description',
                'Decision Result': 'Decision result',
                'Result': 'Decision result',
                'Status': 'Decision result'
            }
            df_temp = df_temp.rename(columns=rename_map)
            
            if 'Full_Description' in df_temp.columns:
                all_dfs.append(df_temp)
                
        except Exception as e:
            continue

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        
        search_col = 'Full_Description'
        cols_display = ['No of HoldTag', 'Customer', 'Cables type', 'Decision result', 'Date', 'Recheck']
        
        for col in cols_display:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '-')
        
        if search_col in df.columns:
            df[search_col] = df[search_col].fillna("ไม่ระบุ")
        else:
            df[search_col] = "ไม่ระบุ"
            
        return df
    
    return None

df = load_data()

# --- 3. ส่วนแสดงผล ---
if df is not None:
    search_col = 'Full_Description'
    result_col = 'Decision result'
    
    # 🔥 [ฟีเจอร์ใหม่ 1] พจนานุกรมคำศัพท์ (แก้ไขตรงนี้ได้เลย) 🔥
    # ระบบจะเอาคำพวกนี้ไปแปะเพิ่มในฐานข้อมูลให้เอง ทำให้ค้นหาเจอทั้งตัวย่อและตัวเต็ม
    synonyms = {
        "T/S": "Tensile Strength แรงดึง",
        "B/S": "Breaking Strength แรงดึงขาด",
        "MG": "Master Batch เม็ดสี",
        "OD": "Outer Diameter ขนาดภายนอก",
        "ID": "Inner Diameter ขนาดภายใน",
        "Elong": "Elongation ยืด",
        # เพิ่มคำอื่นๆ ต่อท้ายได้เลยครับ รูปแบบ "คำค้น": "คำความหมาย"
    }

    # เตรียมข้อมูลสำหรับ AI (รวมคำศัพท์เข้าไปด้วย)
    # ฟังก์ชันช่วยแปลงคำศัพท์
    def enrich_text(text):
        text = str(text).lower()
        for key, value in synonyms.items():
            # ถ้าเจอคำย่อในข้อความ ให้เติมคำเต็มเข้าไปด้วย
            if key.lower() in text:
                text += " " + value
            # หรือถ้าเจอคำเต็ม ก็เติมคำย่อเข้าไป
            if value.lower() in text:
                text += " " + key
        return text

    # สร้างคอลัมน์ใหม่สำหรับให้ AI เรียนรู้ (User ไม่เห็น แต่ AI เห็น)
    df['AI_Search_Text'] = df[search_col].apply(enrich_text)

    # 🔥 [ฟีเจอร์ใหม่ 2] ปรับให้จับคำสั้นได้ดีขึ้น (แก้จาก 3 เป็น 2)
    # ngram_range=(2, 5) -> เจอคำ 2 ตัวอักษรอย่าง MG, OD ได้แล้ว
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(df['AI_Search_Text'])

    st.success(f"✅ ฐานข้อมูลพร้อมใช้งาน ({len(df)} รายการ)")

    top_n = st.slider("จำนวนเคสที่แสดง", 1, 10, 3)
    
    # ช่องค้นหา
    query_input = st.text_input("ระบุปัญหา:", placeholder="เช่น B/S ต่ำ, MG สูง...")

    if query_input:
        # แปลงคำค้นหาด้วยพจนานุกรมเหมือนกัน
        query_enriched = enrich_text(query_input)
        
        query_vec = vectorizer.transform([query_enriched])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-top_n:][::-1]

        st.markdown("---")
        
        # คำนวณ %
        results_to_show = []
        for idx in top_indices:
            score = similarity[idx]
            if score > 0.01: 
                results_to_show.append((idx, score))
        
        total_raw_score = sum([s for _, s in results_to_show])
        
        if not results_to_show:
            st.info("ไม่พบเคสที่คล้ายกัน")
        else:
            for idx, original_score in results_to_show:
                row = df.iloc[idx]
                decision = str(row.get(result_col, '-'))
                
                if total_raw_score > 0:
                    normalized_percent = (original_score / total_raw_score) * 100
                else:
                    normalized_percent = 0
                
                decision_lower = decision.lower()
                status_icon = "🟡"
                msg_func = st.warning
                
                if any(x in decision_lower for x in ['scrap', 'reject', 'ทิ้ง', 'ไม่ผ่าน', 'เสีย']):
                    status_icon = "🔴"
                    msg_func = st.error
                elif any(x in decision_lower for x in ['pass', 'ผ่าน', 'accept', 'ok', 'อนุโลม']):
                    status_icon = "🟢"
                    msg_func = st.success
                
                msg_func(f"{status_icon} {decision} (ความน่าจะเป็น: {normalized_percent:.1f}%)")

                with st.expander(f"ดูรายละเอียด: {row.get('No of HoldTag', '-')}"):
                    st.write(f"**ลูกค้า:** {row.get('Customer', '-')}")
                    st.write(f"**ชนิดสาย:** {row.get('Cables type', '-')}")
                    st.write(f"**อาการ:** {row.get(search_col, '-')}")
                    st.write(f"**วันที่:** {row.get('Date', '-')}")
                    if 'Recheck' in row:
                        st.write(f"**Recheck:** {row.get('Recheck', '-')}")
else:
    st.error("❌ ไม่พบไฟล์ข้อมูล! กรุณาตรวจสอบว่ามีไฟล์ Excel (.xlsx) วางอยู่คู่กับไฟล์ app.py หรือไม่")
