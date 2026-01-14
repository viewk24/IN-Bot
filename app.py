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
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[search_col].astype(str))

    st.success(f"✅ ฐานข้อมูลพร้อมใช้งาน ({len(df)} รายการ)")

    top_n = st.slider("จำนวนเคสที่แสดง", 1, 10, 3)
    query = st.text_input("ระบุปัญหา:", placeholder="เช่น สายมีรอยถลอก...")

    if query:
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-top_n:][::-1]

        st.markdown("---")
        
        # --- [จุดที่แก้ไข] คำนวณ % ให้รวมกันได้ 100 ---
        
        # 1. คัดเลือกเฉพาะเคสที่ผ่านเกณฑ์ (Similarity > 0.01) เก็บไว้ก่อน
        results_to_show = []
        for idx in top_indices:
            score = similarity[idx]
            if score > 0.01:
                results_to_show.append((idx, score))
        
        # 2. หาผลรวมคะแนนดิบทั้งหมด ของเคสที่จะแสดง
        total_raw_score = sum([s for _, s in results_to_show])
        
        if not results_to_show:
            st.info("ไม่พบเคสที่คล้ายกัน")
        else:
            # 3. วนลูปแสดงผล โดยคำนวณ % แบบสัดส่วน (Normalized)
            for idx, original_score in results_to_show:
                row = df.iloc[idx]
                decision = str(row.get(result_col, '-'))
                
                # คำนวณ % ใหม่: (คะแนนดิบ / คะแนนรวม) * 100
                if total_raw_score > 0:
                    normalized_percent = (original_score / total_raw_score) * 100
                else:
                    normalized_percent = 0
                
                # Logic สีเหมือนเดิม
                decision_lower = decision.lower()
                status_icon = "🟡"
                if any(x in decision_lower for x in ['scrap', 'reject', 'ทิ้ง', 'ไม่ผ่าน', 'เสีย']):
                    status_icon = "🔴"
                    msg_func = st.error
                elif any(x in decision_lower for x in ['pass', 'ผ่าน', 'accept', 'ok', 'อนุโลม']):
                    status_icon = "🟢"
                    msg_func = st.success
                else:
                    msg_func = st.warning
                
                # แสดงผล % แบบใหม่
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
