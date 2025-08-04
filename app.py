import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from io import BytesIO
import json
import re

# Load your API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.title("🔄 Program Process Analyzer & Overlap Detector")

uploaded_file = st.file_uploader("Upload Excel Spreadsheet of Programs", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df)

    if st.button("Analyze Programs and Find Process Overlaps"):
        # First, collect all processes for each program
        program_processes = {}
        progress_text = st.empty()
        progress_bar = st.progress(0)

        st.header("📋 Phase 1: Identifying Key Processes")
        
        for i, row in df.iterrows():
            program_name = row['Program']
            program_description = row['Description']
            
            progress_text.text(f"Identifying processes for '{program_name}' ({i+1}/{len(df)})")

            # First prompt: Extract key processes only
            process_prompt = f"""
            Analyze the following government program and identify its KEY PROCESSES.

            Program: {program_name}
            Description: {program_description}

            List the major processes within this program. For each process:
            1. Give it a clear, concise name
            2. Provide a brief description (1-2 sentences)
            3. Identify the process type (e.g., Inspection, Document Review, Application Processing, Data Management, Customer Service, Financial Management, Compliance Monitoring, etc.)

            Format your response as a JSON array like this:
            [
                {{
                    "process_name": "Process Name",
                    "description": "Brief description",
                    "process_type": "Type of process"
                }},
                ...
            ]
            """

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": process_prompt}],
                temperature=0.3
            )

            try:
                # Extract JSON from the response - handle cases where GPT adds extra text
                response_text = response.choices[0].message.content
                
                # Try to find JSON array in the response
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                
                if json_match:
                    json_text = json_match.group()
                    processes = json.loads(json_text)
                else:
                    # Try parsing the entire response as JSON
                    processes = json.loads(response_text)
                
                program_processes[program_name] = processes
                
                # Display processes for this program
                with st.expander(f"Processes for {program_name}"):
                    for proc in processes:
                        st.write(f"**{proc['process_name']}** ({proc['process_type']})")
                        st.write(f"   {proc['description']}")
                
            except (json.JSONDecodeError, KeyError) as e:
                # Fallback: try to extract process information manually
                st.warning(f"JSON parsing failed for {program_name}, using fallback method")
                
                # Simple fallback - just store the raw text
                fallback_processes = [{
                    "process_name": "Process extraction failed",
                    "description": "Unable to parse structured process data. Raw response saved.",
                    "process_type": "Unknown"
                }]
                program_processes[program_name] = fallback_processes
                
                # Still show what we got
                with st.expander(f"Processes for {program_name} (Raw)"):
                    st.text(response.choices[0].message.content)
            
            progress_bar.progress((i+1)/len(df))

        st.success("✅ Process identification complete!")

        # Phase 2: Analyze overlaps
        st.header("🔍 Phase 2: Analyzing Process Overlaps")
        
        # Prepare data for overlap analysis
        all_processes_text = ""
        for program, processes in program_processes.items():
            all_processes_text += f"\n\nProgram: {program}\n"
            for proc in processes:
                all_processes_text += f"- {proc['process_name']} ({proc['process_type']}): {proc['description']}\n"

        overlap_prompt = f"""
        Analyze the following processes from multiple government programs and identify opportunities for consolidation or centralization.

        {all_processes_text}

        Provide a comprehensive analysis including:

        1. **Common Process Types**: Identify process types that appear across multiple programs
        2. **Specific Process Overlaps**: List specific processes that are similar or identical across programs
        3. **Consolidation Opportunities**: Describe specific opportunities to consolidate or centralize processes
        4. **Shared Service Recommendations**: Suggest which processes could become shared services
        5. **Implementation Priority**: Rank consolidation opportunities by potential impact and ease of implementation
        6. **Expected Benefits**: Quantify potential efficiency gains, cost savings, or service improvements

        Be specific about which programs share which processes and how they could be consolidated.
        """

        overlap_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": overlap_prompt}],
            temperature=0.5
        )

        overlap_analysis = overlap_response.choices[0].message.content
        
        st.header("📊 Process Overlap Analysis")
        st.markdown(overlap_analysis)

        # Phase 3: Generate individual program analyses
        st.header("📈 Phase 3: Individual Program Efficiency Analyses")
        
        analyses = []
        progress_text.text("Generating detailed analyses...")
        progress_bar.progress(0)

        for i, row in df.iterrows():
            program_name = row['Program']
            program_description = row['Description']
            
            progress_text.text(f"Analyzing '{program_name}' ({i+1}/{len(df)})")

            # Include the identified processes in the analysis
            processes_text = ""
            if program_name in program_processes:
                processes_text = "\n".join([f"- {p['process_name']}: {p['description']}" for p in program_processes[program_name]])

            analysis_prompt = f"""
            You're analyzing a local government program called '{program_name}' for efficiency and cost savings.

            Program description:
            {program_description}

            Key Processes already identified:
            {processes_text}

            Provide a clearly structured analysis including these sections:

            1. Current State Assessment
            2. Areas to Analyze for Efficiency Opportunities
            3. Key Processes within this Program (use the processes identified above and expand if needed)
            4. Types of Recommendations (Streamlining, Automation, Staffing, Fees, Policy, Customer Service enhancements)
            5. Ideal Efficiency Metrics (suggest measurable metrics for tracking improvements)
            6. Anticipated Outcomes and Benefits

            Clearly organize your response into these sections.
            """

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.5
            )

            analysis = response.choices[0].message.content
            analyses.append({'Program': program_name, 'Analysis': analysis})
            progress_bar.progress((i+1)/len(df))

        st.success("✅ All analyses generated!")

        # Display individual analyses
        st.header("📑 Individual Program Analyses")
        for item in analyses:
            with st.expander(f"Analysis for {item['Program']}"):
                st.markdown(item['Analysis'])

        # Export to Word
        def create_word_doc(program_processes, overlap_analysis, analyses):
            doc = Document()
            doc.add_heading('Program Process Analysis Report', 0)
            
            # Add process overlap analysis first
            doc.add_heading('Process Overlap Analysis', level=1)
            doc.add_paragraph(overlap_analysis)
            doc.add_page_break()
            
            # Add process summary
            doc.add_heading('Process Summary by Program', level=1)
            for program, processes in program_processes.items():
                doc.add_heading(program, level=2)
                for proc in processes:
                    doc.add_paragraph(f"• {proc['process_name']} ({proc['process_type']})")
                    doc.add_paragraph(f"  {proc['description']}", style='Normal')
            doc.add_page_break()
            
            # Add individual analyses
            doc.add_heading('Individual Program Efficiency Analyses', level=1)
            for item in analyses:
                doc.add_heading(item['Program'], level=2)
                doc.add_paragraph(item['Analysis'])
                doc.add_page_break()
            
            bio = BytesIO()
            doc.save(bio)
            bio.seek(0)
            return bio

        word_file = create_word_doc(program_processes, overlap_analysis, analyses)
        st.download_button("📄 Download Complete Word Report", word_file, "Program_Process_Analysis_Report.docx")

        # Export to Excel
        def create_excel_file(program_processes, overlap_analysis, analyses):
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                # Sheet 1: Process Summary
                process_data = []
                for program, processes in program_processes.items():
                    for proc in processes:
                        process_data.append({
                            'Program': program,
                            'Process Name': proc['process_name'],
                            'Process Type': proc['process_type'],
                            'Description': proc['description']
                        })
                
                if process_data:
                    process_df = pd.DataFrame(process_data)
                    process_df.to_excel(writer, index=False, sheet_name='Process Summary')
                
                # Sheet 2: Overlap Analysis
                overlap_df = pd.DataFrame([{'Overlap Analysis': overlap_analysis}])
                overlap_df.to_excel(writer, index=False, sheet_name='Overlap Analysis')
                
                # Sheet 3: Individual Analyses
                analyses_df = pd.DataFrame(analyses)
                analyses_df.to_excel(writer, index=False, sheet_name='Program Analyses')
                
                # Auto-adjust column widths
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 100)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            bio.seek(0)
            return bio

        excel_file = create_excel_file(program_processes, overlap_analysis, analyses)
        st.download_button("📊 Download Complete Excel Report", excel_file, "Program_Process_Analysis_Report.xlsx")