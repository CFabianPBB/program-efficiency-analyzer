import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from io import BytesIO

# Load your API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.title("📈 Program Efficiency Analyzer")

uploaded_file = st.file_uploader("Upload Excel Spreadsheet of Programs", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df)

    if st.button("Generate Efficiency Analyses for All Programs"):
        analyses = []
        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i, row in df.iterrows():
            program_name = row['Program']
            program_description = row['Description']
            
            progress_text.text(f"Analyzing '{program_name}' ({i+1}/{len(df)})")

            prompt = f"""
            You're analyzing a local government program called '{program_name}' for efficiency and cost savings.

            Program description:
            {program_description}

            Provide a clearly structured analysis including these sections:

            1. Current State Assessment
            2. Areas to Analyze for Efficiency Opportunities
            3. Key Processes within this Program (list and briefly describe major processes)
            4. Types of Recommendations (Streamlining, Automation, Staffing, Fees, Policy, Customer Service enhancements)
            5. Ideal Efficiency Metrics (suggest measurable metrics for tracking improvements)
            6. Anticipated Outcomes and Benefits

            Clearly organize your response into these sections.
            """

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )

            analysis = response.choices[0].message.content
            analyses.append({'Program': program_name, 'Analysis': analysis})
            progress_bar.progress((i+1)/len(df))

        st.success("✅ All analyses generated!")

        # Display analyses
        for item in analyses:
            st.subheader(f"Program: {item['Program']}")
            st.markdown(item['Analysis'])

        # Export to Word
        def create_word_doc(analyses):
            doc = Document()
            doc.add_heading('Program Efficiency Analyses', 0)
            for item in analyses:
                doc.add_heading(item['Program'], level=1)
                doc.add_paragraph(item['Analysis'])
                doc.add_page_break()
            bio = BytesIO()
            doc.save(bio)
            bio.seek(0)
            return bio

        word_file = create_word_doc(analyses)
        st.download_button("📄 Download Word Report", word_file, "Program_Efficiency_Analyses.docx")

        # Export to Excel
        def create_excel_file(analyses):
            excel_df = pd.DataFrame(analyses)
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                excel_df.to_excel(writer, index=False, sheet_name='Efficiency Analyses')
            bio.seek(0)
            return bio

        excel_file = create_excel_file(analyses)
        st.download_button("📊 Download Excel Report", excel_file, "Program_Efficiency_Analyses.xlsx")
