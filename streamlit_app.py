__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import math
from pathlib import Path
import os
import json
import re
import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import markdown2
# from IPython.display import Markdown, FileLink, display # Not needed for streamlit
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import google.generativeai as genai
# from kaggle_secrets import UserSecretsClient # Not needed for streamlit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA # Not used in current snippet, but kept from original
from sklearn.cluster import KMeans # Not used in current snippet, but kept from original

# Langchain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader # Used for resume file if uploaded

# Chroma
import sqlite3
import chromadb
from chromadb.config import Settings

# --- Adding font check ---
# Get the absolute path to your font file
font_path = Path(__file__).parent / "fonts" / "NotoColorEmoji.ttf"

# Verify font exists and is accessible
if not font_path.exists():
    st.warning("Font file not found!")

# --- Page Configuration ---
st.set_page_config(
    page_title='🤖 AI Career Coach 💼',
    page_icon='🤖', # Use the emoji directly
    layout='wide'
)

# --- Helper Functions (some adapted from your script) ---
@st.cache_data # Cache data loading
def load_df(uploaded_file):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    return None

@st.cache_data # Cache Gemini API calls
def get_gemini_embeddings(texts, task_type="SEMANTIC_SIMILARITY"):
    # This function will call the API, ensure API key is configured before calling
    return [genai.embed_content(model="models/text-embedding-004", content=t, task_type=task_type)['embedding'] for t in texts]

@st.cache_data(show_spinner=False) # Cache Gemini API calls for content generation
def generate_gemini_content(model_name, prompt_text):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt_text)
    return response.text

def extract_json_from_text(text):
    # Try to find JSON within backticks
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_string = match.group(1)
    else:
        # If not found in backticks, try to find any valid JSON object
        # This is a bit more lenient but might grab unintended JSON if the text is complex
        match = re.search(r'(\{[\s\S]*\})', text)
        if match:
            json_string = match.group(1)
        else:
            st.error("No JSON block found in the model's response.")
            return {} # Return empty dict if no JSON found

    # Remove non-printable ASCII control characters (excluding tab, newline, carriage return)
    cleaned_json_string = ''.join(char for char in json_string if 32 <= ord(char) < 127 or ord(char) in [9, 10, 13])
    try:
        return json.loads(cleaned_json_string)
    except json.JSONDecodeError as e:
        st.error(f"JSONDecodeError after cleaning: {e}")
        st.text_area("Problematic JSON string:", cleaned_json_string, height=200)
        return {} # Return an empty dict or handle the error as needed

# --- API Key Configuration ---
# Try to get API key from Streamlit secrets (for deployment)
try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key: # If key exists in secrets but is empty
        api_key = None
except AttributeError: # If st.secrets is not available (local development without secrets.toml)
    api_key = None

if not api_key:
    st.sidebar.subheader("🔑 API Key Configuration")
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password", help="Get your key from AI Studio.")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("API Key Configured!")
    except Exception as e:
        st.sidebar.error(f"API Key Configuration Error: {e}")
        st.stop()
else:
    st.warning("Please enter your Google API Key in the sidebar to use the app.")
    st.stop()


# --- Main App ---
st.markdown('''
# 🤖 AI Career Coach for Busy Data Professionals 💼

A GenAI-powered assistant to analyze job trends, extract skills, and provide personalized career advice.
Finding a clear career path in the evolving data industry can be overwhelming.
This app leverages Generative AI to help data professionals navigate their career journey using real job market data, skill extraction, and personalized role recommendations.
Unlike other comparison tools, this app takes it one step further - by providing next step guidance to candidates looking for insight on potential areas to improve and actions they can take today to align themselves better to role and job description requirements.
''')

st.markdown('''
---
## ⚛ GenAI Capabilities Used
- **Document Understanding:** Parse job descriptions to extract in-demand skills.
- **Embeddings + Vector Search:** Match a user's resume to the most relevant job clusters.
- **Few-shot Prompting + Structured Output:** Generate tailored career advice with recommendations.

Models used (via Vertex AI / Google AI Studio):
- `gemini-2.0-flash-lite` (or similar for skill extraction)
- `text-embedding-004`
- `gemini-2.0-flash` (or latest available free pro model)
''')

st.markdown('''
---
## 📌 Setup & Inputs
''')

# --- Inputs ---
# Define paths to sample data
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data"
SAMPLE_RESUME = SAMPLE_DATA_PATH / "da_resume.txt"
SAMPLE_JOBS = SAMPLE_DATA_PATH / "data_jobs_50.csv"

# Create two columns for data source selection
col1, col2 = st.columns(2)

with col1:
    jobs_data_source = st.radio(
        "Choose jobs data source:",
        ["Upload Jobs CSV", "Use Sample Jobs Data"],
        help="You can either upload your own jobs data or use our sample dataset"
    )

with col2:
    resume_data_source = st.radio(
        "Choose resume source:",
        ["Paste Resume", "Use Sample Resume"],
        help="You can either paste your own resume or use our sample resume"
    )

# Handle jobs data selection
if jobs_data_source == "Upload Jobs CSV":
    uploaded_job_data_file = st.file_uploader(
        "1. Upload your job data CSV",
        type="csv",
        help="A small .csv dataset (e.g., 50 rows) of job data that must include a column named 'description' and optionally, a column named 'title' to use for analysis."
    )
else:
    uploaded_job_data_file = SAMPLE_JOBS
    with st.expander("Preview sample jobs data"):
        try:
            sample_df = pd.read_csv(SAMPLE_JOBS)
            st.dataframe(sample_df.head())
            st.success("✅ Using sample jobs data")
        except Exception as e:
            st.error(f"Error loading sample jobs data: {e}")

# Handle resume selection
if resume_data_source == "Paste Resume":
    resume_text_input = st.text_area(
        "2. Paste your resume text here:",
        height=200,
        placeholder="No personal details are needed - just experience, skills, etc.",
        help="Paste your anonymized resume content."
    )
else:
    try:
        with open(SAMPLE_RESUME, 'r') as f:
            resume_text_input = f.read()
        with st.expander("Preview sample resume"):
            st.text(resume_text_input)
            st.success("✅ Using sample resume")
    except Exception as e:
        st.error(f"Error loading sample resume: {e}")
        resume_text_input = ""

# Analyze button and validation
if st.button("✨ Analyze Career Path"):
    # Validate jobs data
    if jobs_data_source == "Upload Jobs CSV" and not uploaded_job_data_file:
        st.error("❌ Please upload the job data CSV file.")
        st.stop()
    
    # Validate resume
    if resume_data_source == "Paste Resume" and not resume_text_input.strip():
        st.error("❌ Please paste your resume text.")
        st.stop()
    
    # Load and validate jobs data
    if isinstance(uploaded_job_data_file, Path):
        df = pd.read_csv(uploaded_job_data_file)  # For sample data
    else:
        df = load_df(uploaded_job_data_file)  # For uploaded data
        
    if df is None or 'description' not in df.columns:
        st.error("❌ CSV file is invalid or does not contain a 'description' column.")
        st.stop()

    job_descriptions = df['description'].dropna().tolist()
    if not job_descriptions:
        st.error("❌ No job descriptions found in the file after dropping NaNs.")
        st.stop()

    # Limit to 50 for this demo
    job_descriptions = job_descriptions[:50]
    st.info(f"Found {len(df)} jobs in CSV. Analyzing up to {len(job_descriptions)} job descriptions.")

    # --- ⚗️ GenAI Capability 1: Document Understanding — Skill Extraction ---
    st.markdown("--- \n## ⚗️ GenAI Capability 1: Document Understanding — Skill Extraction")
    
    # For Streamlit, we might not want to save to a file by default unless explicitly needed
    # results_file = "extracted_skills.json" # os.path.join("/kaggle/working/", "extracted_skills.json")
    # For simplicity in Streamlit, let's re-extract each time or use st.cache for the extraction block
    
    all_hard_skills, all_soft_skills, results_list = [], [], []
    model_skill_extraction = genai.GenerativeModel('models/gemini-2.0-flash-lite') # or a current equivalent

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Extracting skills from job descriptions... This may take a moment."):
        for i, job_desc_text in enumerate(job_descriptions):
            status_text.text(f"🔄 Processing job {i+1}/{len(job_descriptions)}...")
            prompt = f"""
            Extract the hard and soft skills from the following job description.
            Format as JSON with 'Hard Skills' and 'Soft Skills' keys.
            Each skill should be a concise phrase.

            Job Description:
            {job_desc_text}
            """
            try:
                # Using the cached function for API call if this part is refactored into one
                response_text = generate_gemini_content('models/gemini-2.0-flash-lite', prompt) # Adjust model if needed
                # text = re.sub(r"```json|```", "", response.text.strip()) # Handled in extract_json
                skills = extract_json_from_text(response_text)
                
                hard = skills.get('Hard Skills', [])
                soft = skills.get('Soft Skills', [])
                
                all_hard_skills.extend(s for s in hard if isinstance(s, str)) # Ensure skills are strings
                all_soft_skills.extend(s for s in soft if isinstance(s, str))
                results_list.append({"job_index": i, "hard_skills": hard, "soft_skills": soft})

            except Exception as e:
                st.error(f"❌ Error processing job {i}: {e}\nRaw response was: {response_text[:200]}...") # Show partial response
            time.sleep(0.5) # Respect API limits if any; Gemini often handles rate limiting well
            progress_bar.progress((i + 1) / len(job_descriptions))
        status_text.success(f"✅ Skill extraction complete for {len(job_descriptions)} jobs!")

    # --- 📊 Visualize Top Skills ---
    st.markdown("### 📊 Top Skills Visualization")
    def plot_skills_st(skill_data, title, key_suffix):
        if not skill_data:
            st.write(f"No data to plot for {title}")
            return None, None # Return None for fig and filename
        
        # Ensure skills are strings for plotting
        skills_cleaned = [str(item[0]) for item in skill_data]
        counts_cleaned = [item[1] for item in skill_data]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=np.array(counts_cleaned), y=np.array(skills_cleaned), palette="crest", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Skill")
        plt.tight_layout()
        
        # Save figure to a BytesIO object for download
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        
        return fig, img_bytes

    from io import BytesIO

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Hard Skills")
        top_hard = Counter(all_hard_skills).most_common(10)
        if top_hard:
            fig_hard, img_hard_bytes = plot_skills_st(top_hard, "Top 10 Hard Skills", "hard")
            if fig_hard:
                st.pyplot(fig_hard)
                st.download_button(
                    label="Download Hard Skills Chart",
                    data=img_hard_bytes,
                    file_name="top_hard_skills.png",
                    mime="image/png"
                )
            for s, c in top_hard: st.write(f"  {s}: {c}")
        else:
            st.write("No hard skills extracted.")

    with col2:
        st.subheader("Top 10 Soft Skills")
        top_soft = Counter(all_soft_skills).most_common(10)
        if top_soft:
            fig_soft, img_soft_bytes = plot_skills_st(top_soft, "Top 10 Soft Skills", "soft")
            if fig_soft:
                st.pyplot(fig_soft)
                st.download_button(
                    label="Download Soft Skills Chart",
                    data=img_soft_bytes,
                    file_name="top_soft_skills.png",
                    mime="image/png"
                )
            for s, c in top_soft: st.write(f"  {s}: {c}")
        else:
            st.write("No soft skills extracted.")


    # --- 𝄂𝄂𝄀𝄁 GenAI Capability 2: Embeddings + Vector Search 👀 ---
    st.markdown("--- \n## 💎 GenAI Capability 2: Embeddings + Vector Search")
    with st.spinner("Generating embeddings and performing vector search..."):
        # Resume Embedding (using the text from st.text_area)
        resume_embedding = get_gemini_embeddings([resume_text_input])[0]

        # Prepare & Chunk Job Descriptions
        documents = [Document(page_content=d) for d in job_descriptions] # Use the filtered list
        docs_split = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

        all_doc_texts = [doc.page_content for doc in docs_split]
        all_doc_embeddings = get_gemini_embeddings(all_doc_texts)

        # ChromaDB
        # Use a relative path for Streamlit; this DB will be ephemeral on Streamlit Cloud
        chroma_db_path = "./chroma_db_career_coach"
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        collection_name = "job_descriptions_career_coach"
        
        # Delete collection if it exists to start fresh each time for this demo
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass # Collection might not exist, which is fine
            
        collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        if docs_split: # Ensure there are documents to add
            collection.add(
                ids=[f"doc_{i}" for i in range(len(docs_split))],
                documents=all_doc_texts,
                embeddings=all_doc_embeddings,
                metadatas=[{"source": f"job_desc_chunk_{i}"} for i in range(len(docs_split))] # More descriptive metadata
            )
            st.success(f"📚 Added {len(docs_split)} document chunks to ChromaDB.")

            # Query Resume vs Jobs
            st.subheader("🔎 Top 3 Job Matches from Vector Search")
            query_results = collection.query(query_embeddings=[resume_embedding], n_results=min(3, len(docs_split)))
            
            if query_results and query_results["documents"] and query_results["documents"][0]:
                for i, (doc_text, dist) in enumerate(zip(query_results["documents"][0], query_results["distances"][0])):
                    similarity = 1 - dist
                    bar = "🟩" * int(similarity * 10) + "⬜" * (10 - int(similarity * 10))
                    st.markdown(f"**Match #{i + 1}** ({bar} {int(similarity * 100)}% match, Cosine Distance: {dist:.4f})")
            
                    # Try to get the job title from the original DataFrame
                    try:
                        job_index = int(query_results["metadatas"][0][i]["source"].split("_")[-1])  # Extract original index
                        if 'title' in df.columns:
                            job_title = df.iloc[job_index]['title']
                            # Truncate title to 30 characters
                            truncated_title = textwrap.shorten(str(job_title), width=30, placeholder="...")
                            st.markdown(f"**{truncated_title}**:  > {textwrap.shorten(doc_text, width=500, placeholder='...')}")
                        else:
                            st.markdown(f"> {textwrap.shorten(doc_text, width=500, placeholder='...')}")
                    except (KeyError, ValueError, IndexError) as e:
                        st.warning(f"Could not retrieve job title: {e}. Displaying description only.")
                        st.markdown(f"> {textwrap.shorten(doc_text, width=500, placeholder='...')}")
            
            else:
                st.warning("No matching documents found in vector search.")
        else:
            st.warning("No document chunks were created for ChromaDB. Skipping vector search.")

        # Cosine Similarity Heatmap (Resume vs Top N Job Chunks)
        st.subheader("🔥 Cosine Similarity Heatmap: Resume vs Top Job Chunks")
        top_n_heatmap = min(10, len(docs_split))  # Show up to 10 or fewer if not enough docs
        if top_n_heatmap > 0:
            # Using all_doc_embeddings which are already generated for all chunks
            job_chunk_embeddings_for_heatmap = all_doc_embeddings[:top_n_heatmap]
        
            similarities = cosine_similarity([resume_embedding], job_chunk_embeddings_for_heatmap)[0]
            # Get job titles for the heatmap labels
            heatmap_labels = []
            try:
                if 'title' in df.columns:
                    for i in range(top_n_heatmap):
                        try:
                            # Match the metadata access pattern from your working Top 3 code
                            source_info = docs_split[i].metadata["source"]  # Direct access like query_results
                            job_index = int(source_info.split("_")[-1])
                            job_title = df.iloc[job_index]['title']
                            truncated_title = textwrap.shorten(str(job_title), width=30, placeholder="...")
                            heatmap_labels.append(truncated_title)
                        except (KeyError, ValueError, IndexError) as e:
                            # st.write(f"Debug - Error for chunk {i}:", e)  # Optional debug line
                            heatmap_labels.append(f"Job Chunk {i + 1}")
                else:
                    heatmap_labels = [f"Job Chunk {i + 1}" for i in range(top_n_heatmap)]
            except Exception as e:
                st.warning(f"Error retrieving heatmap titles: {e}. Using chunk numbers.")
                heatmap_labels = [f"Job Chunk {i + 1}" for i in range(top_n_heatmap)]
            
            # Optional debug line to see what labels were generated
            # st.write("Debug - Heatmap labels:", heatmap_labels)
            
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 2))
            sns.heatmap(similarities.reshape(1, -1), annot=True, cmap="YlGnBu",
                        xticklabels=heatmap_labels,
                        yticklabels=["Resume"],
                        vmin=0, vmax=1, ax=ax_heatmap)
            
            ax_heatmap.set_title(f"Cosine Similarity: Resume vs Top {top_n_heatmap} Job Description Chunks")
            plt.tight_layout()
            st.pyplot(fig_heatmap)
        
            img_heatmap_bytes = BytesIO()
            plt.savefig(img_heatmap_bytes, format='png')
            img_heatmap_bytes.seek(0)
            st.download_button(
                label="Download Heatmap",
                data=img_heatmap_bytes,
                file_name=f"similarity_heatmap_top_{top_n_heatmap}.png",
                mime="image/png"
            )
        else:
            st.write("Not enough job chunks to create a heatmap.")

    # --- 🗺️ GenAI Capability 3: Career Advice ---
    st.markdown("--- \n## 🗺️ GenAI Capability 3: Personalized Career Guidance")
    with st.spinner("Generating personalized career advice... This may take a couple of calls to the AI."):
        # Using a current, robust model. You might need to adjust model names.
        # Consider using a single, more capable model for simplicity if desired.
        model_name_pro = 'gemini-2.0-flash'

        # 1, 3, 4 from one model call
        prompt_1_3_4 = f"""
        You are an expert career coach for data professionals.
        Given the following resume, provide a detailed, friendly, and personalized career path analysis.

        Format your response STRICTLY as a JSON object with the following keys:
        - "Current Role": A single sentence summarizing the candidate's likely current role or experience level.
        - "Missing Skills": A list of objects. Each object should have:
            - "Skill": (string) The name of the skill.
            - "Details": (string) Why this skill is important for career progression, tailored to the resume.
            - "Relevance": (string) How it connects to potential next roles.
        - "Learning Resources": A list of objects. Each object should have:
            - "Area": (string) The skill or area to focus on.
            - "ResourceSuggestion": (string) Suggest specific free or low-cost online resources (e.g., a type of course on Coursera, a specific YouTube channel, official documentation).

        Resume:
        ---
        {resume_text_input}
        ---
        Ensure the output is only the JSON object, starting with {{ and ending with }}.
        """
        response_1_3_4_text = generate_gemini_content(model_name_pro, prompt_1_3_4)
        part_1_3_4_json = extract_json_from_text(response_1_3_4_text)
        if not part_1_3_4_json:
            st.error("Failed to get Current Role, Missing Skills, or Learning Resources. Model output:")
            st.code(response_1_3_4_text)


        # 2 from another model call (or combine into one prompt)
        prompt_2 = f"""
        You are an expert career coach for data professionals.
        Based on the provided resume, suggest potential next career roles.

        Format your response STRICTLY as a JSON object with the key "Suggested Next Roles".
        "Suggested Next Roles" should be a list of objects. Each object should have:
        - "Role": (string) The title of the suggested role.
        - "Reasoning": (string) Why this role is a good fit or next step based on the resume.

        Resume:
        ---
        {resume_text_input}
        ---
        Ensure the output is only the JSON object, starting with {{ and ending with }}.
        """
        response_2_text = generate_gemini_content(model_name_pro, prompt_2) # Using same pro model
        part_2_json = extract_json_from_text(response_2_text)
        if not part_2_json:
            st.error("Failed to get Suggested Next Roles. Model output:")
            st.code(response_2_text)

        # Merge JSONs
        career_advice = {
            "Current Role": part_1_3_4_json.get("Current Role", "N/A"),
            "Suggested Next Roles": part_2_json.get("Suggested Next Roles", []),
            "Missing Skills": part_1_3_4_json.get("Missing Skills", []),
            "Learning Resources": part_1_3_4_json.get("Learning Resources", [])
        }

        # Format Suggested Next Roles
        next_roles_md_list = []
        for item in career_advice.get("Suggested Next Roles", []):
            if isinstance(item, dict):
                next_roles_md_list.append(f"- **{item.get('Role', 'N/A')}**: {item.get('Reasoning', 'N/A')}")
            elif isinstance(item, str):
                 next_roles_md_list.append(f"- {item}")
        next_roles_md = "\n".join(next_roles_md_list) if next_roles_md_list else "*No specific next roles suggested.*"


        # Format Missing Skills
        missing_skills_md_list = []
        for item in career_advice.get("Missing Skills", []):
            if isinstance(item, dict):
                missing_skills_md_list.append(f"- **{item.get('Skill', item.get('Details', 'N/A'))}**: {item.get('Details', item.get('Relevance', 'N/A'))} ({item.get('Relevance', 'Relevant for progression')})")
            elif isinstance(item, str):
                missing_skills_md_list.append(f"- {item}")
        missing_skills_md = "\n".join(missing_skills_md_list) if missing_skills_md_list else "*No specific missing skills identified or already well-rounded!*"


        # Format Learning Resources
        learning_resources_md_list = []
        for item in career_advice.get("Learning Resources", []):
            if isinstance(item, dict):
                learning_resources_md_list.append(f"- **{item.get('Area', 'N/A')}**: {item.get('ResourceSuggestion', 'N/A')}")
            elif isinstance(item, str):
                learning_resources_md_list.append(f"- {item}")
        learning_resources_md = "\n".join(learning_resources_md_list) if learning_resources_md_list else "*No specific learning resources suggested.*"

        md_output = f"""
### 🎯 AI Career Coach: Personalized Career Advice

**✅ Current Role:** {career_advice.get("Current Role", "N/A")}

**🔼 Suggested Next Roles:**
{next_roles_md}

**🧩 Missing Skills to Develop:**
{missing_skills_md}

**📚 Learning Resources & Suggestions:**
{learning_resources_md}
        """
        st.markdown(md_output)

        # --- PDF Generation (Requires NotoColorEmoji.ttf in the same directory or correct path) ---
        try:
            html_body_content = markdown2.markdown(md_output)
            
            # Set up font configuration
            font_config = FontConfiguration()
            font_path = Path(__file__).parent / "fonts" / "NotoColorEmoji.ttf"
            
            # Create CSS with font configuration
            css = CSS(string=f'''
                @page {{ margin: 20px; }}
                @font-face {{
                    font-family: "Noto Color Emoji";
                    src: url("{font_path}") format("truetype");
                }}
                body {{
                    font-family: "Noto Color Emoji", "Helvetica", "Arial", sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 1rem auto;
                    padding: 20px;
                }}
            ''', font_config=font_config)
        
            html_full_for_pdf = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                {html_body_content}
            </body>
            </html>
            """
        
            # Create PDF with proper configuration
            pdf_bytes = HTML(string=html_full_for_pdf).write_pdf(
                stylesheets=[css],
                font_config=font_config
            )
            
            st.download_button(
                label="📥 Download Career Advice PDF",
                data=pdf_bytes,
                file_name="ai_career_coach_advice.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
            st.info("You can copy the Markdown text above as an alternative.")

    st.balloons()
    st.success("🎉 Analysis Complete!")


st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io) & Google Generative AI Models.")
st.markdown("[Adapted from an original Kaggle Notebook concept.](https://www.kaggle.com/code/veronicazaitoun/ai-career-coach/notebook)")
