import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from googletrans.client import Translator
import tempfile
import os
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from gtts import gTTS
import base64
import json
from datetime import datetime
import plotly.express as px
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from typing import Optional
import numpy as np
import plotly.graph_objects as go

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set page config
st.set_page_config(
    page_title="Document Abstracter",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div>div>select {
        background-color: #f0f2f6;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #e8f5e9;
        padding: 2px 5px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'abstract' not in st.session_state:
    st.session_state.abstract = None
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = {}
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = {}
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_abstract' not in st.session_state:
    st.session_state.current_abstract = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

# Initialize models and tools
@st.cache_resource
def load_summarizer():
    try:
        # Using T5-large for better quality summaries
        model_name = "t5-base"
        
        # Add memory-efficient configurations
        torch.cuda.empty_cache()  # Clear GPU memory if available
        
        # Configure model loading with memory-efficient settings
        model_config = {
            "force_download": False,
            "local_files_only": False,
            "low_cpu_mem_usage": True,
            "device_map": "auto"  # Automatically handle device placement
        }
        
        # Load tokenizer with memory-efficient settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **model_config
        )
        
        # Load model with memory-efficient settings
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            **model_config
        )
        
        # Enable memory-efficient inference
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            # Test model with a small input
            test_input = tokenizer("test", return_tensors="pt", max_length=512, truncation=True)
            model.generate(test_input["input_ids"], max_length=50)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")
        return None, None

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

def extract_text_from_file(file):
    """Extract text from uploaded file based on its type."""
    file_extension = Path(file.name).suffix.lower()
    
    try:
        if file_extension == '.txt':
            text = file.getvalue().decode('utf-8')
            # Clean up text
            text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
            return text.strip()
            
        elif file_extension == '.pdf':
            doc = fitz.open(stream=file.getvalue(), filetype="pdf")
            text = ""
            for page in doc:
                # Extract text with better formatting
                text += page.get_text("text") + "\n"
            doc.close()
            # Clean up text
            text = re.sub(r'\r\n', '\n', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()
            
        elif file_extension == '.docx':
            doc = Document(file)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    text.append(paragraph.text.strip())
            return "\n\n".join(text)
            
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        raise

def preprocess_text(text):
    """Preprocess text for better summarization."""
    try:
        # Normalize line endings and whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Clean and normalize text
        cleaned_sentences = []
        for sentence in sentences:
            # Remove extra whitespace and special characters while preserving important punctuation
            sentence = re.sub(r'[^\w\s.,!?-]', ' ', sentence)
            sentence = ' '.join(sentence.split())
            # Only keep meaningful sentences with more than 5 words and less than 100 words
            if 5 < len(sentence.split()) < 100:
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    except Exception as e:
        st.error(f"Error preprocessing text: {str(e)}")
        return text

def generate_abstract(text, length="medium"):
    """Generate abstractive summary using T5-large model with improved processing."""
    try:
        # Add logging
        st.info("Loading summarization model...")
        model, tokenizer = load_summarizer()
        if model is None or tokenizer is None:
            st.error("Failed to load the summarization model. Please try again.")
            return "Error: Could not load the summarization model.", []

        # Preprocess text
        st.info("Preprocessing text...")
        processed_text = preprocess_text(text)
        if not processed_text:
            st.error("No valid text to process after preprocessing.")
            return "Error: No valid text to process.", []
        
        # Split text into smaller chunks for faster processing
        max_chunk_length = 512  # T5's optimal chunk size
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split into sentences first
        try:
            sentences = sent_tokenize(processed_text)
        except Exception as e:
            st.error(f"Error tokenizing sentences: {str(e)}")
            return "Error: Could not process text sentences.", []
        
        # Improved chunking strategy
        for sentence in sentences:
            try:
                sentence_length = len(tokenizer.encode(sentence))
                if current_length + sentence_length > max_chunk_length:
                    if current_chunk:  # Only add non-empty chunks
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            except Exception as e:
                st.warning(f"Skipping problematic sentence: {str(e)}")
                continue
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        if not chunks:
            st.error("No valid text chunks to process.")
            return "Error: No valid text chunks to process.", []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate summaries for each chunk
        summaries = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            try:
                # Update progress
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                
                # Update status with estimated time
                estimated_time = {
                    "short": 10,
                    "medium": 20,
                    "long": 30
                }[length]
                remaining_time = estimated_time * (1 - progress)
                status_text.text(f"Processing chunk {i+1}/{total_chunks}... Estimated time remaining: {remaining_time:.1f} seconds")
                
                # Adjust max_length based on user preference
                max_length = {
                    "short": 150,  # T5's optimal lengths
                    "medium": 250,
                    "long": 350
                }[length]
                
                # Prepare input with T5-specific format
                input_text = f"summarize: {chunk}"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                # Generate summary with optimized parameters for T5
                with torch.no_grad():  # Disable gradient computation
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        min_length=40,  # T5's optimal minimum length
                        num_beams=4,    # T5's optimal beam size
                        length_penalty=1.0,  # T5's optimal length penalty
                        early_stopping=True,
                        do_sample=True,  # Enable sampling for more natural text
                        temperature=0.7,  # T5's optimal temperature
                        top_p=0.9,       # Nucleus sampling
                        repetition_penalty=1.2
                    )
                
                # Decode and clean up the summary
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                if summary.strip():  # Only add non-empty summaries
                    summaries.append(summary)
                    
                # Clear memory after each chunk
                torch.cuda.empty_cache()  # Clear GPU memory if available
                
            except Exception as e:
                st.warning(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if not summaries:
            st.error("No valid summaries were generated.")
            return "Error: Could not generate valid summaries.", []
        
        # Combine summaries and remove duplicates
        final_summary = " ".join(summaries)
        sentences = sent_tokenize(final_summary)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            try:
                sentence = re.sub(r'[^\w\s.,!?-]', ' ', sentence)
                sentence = ' '.join(sentence.split())
                # Improved sentence filtering
                if (sentence not in seen and 
                    len(sentence.split()) > 5 and  # T5's optimal minimum sentence length
                    not any(sentence in other for other in seen)):
                    unique_sentences.append(sentence)
                    seen.add(sentence)
            except Exception as e:
                st.warning(f"Skipping problematic sentence: {str(e)}")
                continue
        
        if not unique_sentences:
            st.error("No valid sentences in the final summary.")
            return "Error: Could not generate valid summary sentences.", []
        
        final_summary = " ".join(unique_sentences)
        
        # Post-process the summary
        try:
            nlp = load_spacy()
            doc = nlp(final_summary)
            
            # Improved key phrase extraction
            key_phrases = []
            for chunk in doc.noun_chunks:
                # Only include meaningful phrases
                if len(chunk.text.split()) >= 2 and not any(word.is_stop for word in chunk):
                    key_phrases.append(chunk.text)
        except Exception as e:
            st.warning(f"Error extracting key phrases: {str(e)}")
            key_phrases = []
        
        return final_summary, key_phrases
    except Exception as e:
        st.error(f"Error generating abstract: {str(e)}")
        return "Error: Could not generate abstract. Please try again.", []

def analyze_text(text):
    """Analyze text for various metrics and insights."""
    nlp = load_spacy()
    doc = nlp(text)
    
    # Word frequency analysis with lemmatization
    tokens_filtered = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            # Lemmatize the token and convert to lowercase
            lemma = token.lemma_.lower()
            if len(lemma) > 2:  # Only keep lemmas longer than 2 characters
                tokens_filtered.append(lemma)
    
    # Create word frequency dictionary from lemmatized tokens
    word_freq = Counter(tokens_filtered)
    
    # Named entity recognition with categorization
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:  # Avoid duplicates
            entities[ent.label_].append(ent.text)
    
    # Calculate readability metrics
    sentences = list(doc.sents)
    words_per_sentence = [len(sent) for sent in sentences]
    avg_sentence_length = sum(words_per_sentence) / len(sentences) if sentences else 0
    
    # Calculate word length distribution
    word_lengths = [len(word) for word in tokens_filtered]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    # Calculate part-of-speech distribution
    pos_counts = Counter([token.pos_ for token in doc])
    
    # Calculate unique word ratio
    unique_words = len(set(tokens_filtered))
    total_words = len(tokens_filtered)
    unique_word_ratio = unique_words / total_words if total_words > 0 else 0
    
    # Calculate additional metrics
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    verb_phrases = [token.text for token in doc if token.pos_ == "VERB"]
    
    return {
        'word_frequency': dict(word_freq.most_common(10)),
        'lemmatized_tokens': tokens_filtered,
        'entities': entities,
        'readability_metrics': {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio
        },
        'pos_distribution': dict(pos_counts),
        'sentence_count': len(sentences),
        'word_count': total_words,
        'unique_word_count': unique_words,
        'noun_phrases': noun_phrases,
        'verb_phrases': verb_phrases
    }

def generate_wordcloud(tokens):
    """Generate word cloud from lemmatized tokens."""
    # Create word cloud with improved settings
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='coolwarm',
        max_words=100,
        min_font_size=10,
        max_font_size=100,
        random_state=42,
        prefer_horizontal=0.7,
        collocations=False
    ).generate(" ".join(tokens))
    
    # Create figure with improved styling
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Most Frequent Lemmatized Words", pad=20, fontsize=16, color='white')
    plt.tight_layout(pad=0)
    
    # Set figure background to black
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    return plt.gcf()

def download_abstract(abstract, filename="abstract.txt"):
    """Create a download button for the abstract."""
    buffer = io.StringIO()
    buffer.write(abstract)
    buffer.seek(0)
    return buffer.getvalue()

def translate_text(text, target_lang):
    """Translate text to target language using Google Translate."""
    try:
        translator = Translator()
        # Split text into smaller chunks for better translation
        max_chunk_length = 5000
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        translated_chunks = []
        for chunk in chunks:
            translation = translator.translate(chunk, dest=target_lang)
            translated_chunks.append(translation.text)
        
        return " ".join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return "Error: Could not translate the text. Please try again."

def text_to_speech(text, lang, output_file):
    """Convert text to speech using gTTS."""
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)

def get_audio_player(audio_file):
    """Create an HTML audio player for the audio file."""
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

def save_to_history(text, abstract, analysis):
    """Save the current session to history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        'timestamp': timestamp,
        'text': text[:200] + '...',  # Store first 200 chars
        'abstract': abstract,
        'analysis': analysis
    })

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app helps you:
    - 📝 Generate abstractive summaries
    - 📊 Analyze text content
    - 🌐 Translate content
    - 🔊 Listen to translations
    - 📈 View text analytics
    """)
    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("""
    1. Upload any text document
    2. Choose summary length
    3. Get your abstract
    4. View text analysis
    5. Translate to any language
    6. Listen to the translation
    """)
    
 
    
    # Add analysis selection in sidebar
    if 'analysis' in st.session_state and st.session_state.analysis:
        st.markdown("### 📊 Analysis Options")
        analysis_options = {
            "Word Frequency": "word_frequency",
            "Word Cloud": "wordcloud",
            "Part of Speech": "pos_distribution",
            "Named Entities": "entities",
            "Readability Metrics": "readability_metrics",
            "Word Statistics": "word_stats"
        }
        
        # Add select all option
        if st.checkbox("Select All Analyses", key="select_all"):
            st.session_state.selected_analyses = list(analysis_options.keys())
        else:
            selected_analyses = st.multiselect(
                "Choose which analyses to display",
                options=list(analysis_options.keys()),
                default=["Word Frequency", "Word Cloud", "Part of Speech", "Named Entities", "Readability Metrics", "Word Statistics"]
            )
            st.session_state.selected_analyses = selected_analyses
        
        # Show selected count
        if 'selected_analyses' in st.session_state:
            st.markdown(f"**Selected Analyses:** {len(st.session_state.selected_analyses)}/{len(analysis_options)}")
    
       # Add clear history button
    if st.session_state.history:
        if st.button("🗑️ Clear History", type="primary"):
            st.session_state.history = []
            st.success("History cleared successfully!")

    # Show history
    if st.session_state.history:
        st.markdown("### 📚 History")
        for item in reversed(st.session_state.history):
            with st.expander(f"📄 {item['timestamp']}", expanded=False):
                st.markdown("**Original Text:**")
                st.text(item['text'])
                st.markdown("**Abstract:**")
                st.markdown(f'''
                <div style="background-color:rgb(0, 0, 0); padding: 15px; border-radius: 5px;">
                    {item['abstract']}
                </div>
                ''', unsafe_allow_html=True)
                if item['analysis']:
                    st.markdown("**Analysis:**")
                    st.json(item['analysis'])

# Main UI
st.title("📚 Document Abstracter")
st.write("Upload a document to get an abstractive summary, analyze content, and translate it to multiple languages!")

# File upload with drag and drop
uploaded_file = st.file_uploader(
    "Upload a file (TXT, PDF, or DOCX)",
    type=['txt', 'pdf', 'docx'],
    help="Drag and drop your file here or click to browse"
)

if uploaded_file:
    try:
        with st.spinner("Processing document..."):
            text = extract_text_from_file(uploaded_file)
            
            # Show document statistics in a nice card
            st.subheader("📊 Document Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", len(text.split()))
            with col2:
                st.metric("Sentences", len(sent_tokenize(text)))
            with col3:
                st.metric("Characters", len(text))
            
            # Show original text in expandable section with better formatting
            with st.expander("📄 View Original Text", expanded=False):
                st.text_area("Original Text", text, height=200, label_visibility="collapsed")
        
        # Text Analysis
        st.subheader("📈 Text Analysis")
        if st.button("Analyze Text", key="analyze"):
            with st.spinner("Analyzing text..."):
                analysis = analyze_text(text)
                st.session_state.analysis = analysis
                
                # Display selected analyses in two columns
                if 'selected_analyses' in st.session_state:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for analysis_name in st.session_state.selected_analyses[:len(st.session_state.selected_analyses)//2]:
                            if analysis_name == "Word Frequency":
                                st.markdown("### Word Frequency")
                                # 2D bar plot
                                fig = px.bar(
                                    x=list(analysis['word_frequency'].keys()),
                                    y=list(analysis['word_frequency'].values()),
                                    title="Top 10 Most Common Lemmatized Words",
                                    color=list(analysis['word_frequency'].values()),
                                    color_continuous_scale='Viridis'
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='black')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif analysis_name == "Word Cloud":
                                st.markdown("### Word Cloud")
                                wordcloud_fig = generate_wordcloud(analysis['lemmatized_tokens'])
                                st.pyplot(wordcloud_fig, use_container_width=True)
                            
                            elif analysis_name == "Part of Speech":
                                st.markdown("### Part of Speech Distribution")
                                # 2D pie chart
                                pos_fig = px.pie(
                                    values=list(analysis['pos_distribution'].values()),
                                    names=list(analysis['pos_distribution'].keys()),
                                    title="Distribution of Parts of Speech"
                                )
                                st.plotly_chart(pos_fig, use_container_width=True)
                    
                    with col2:
                        # Word Statistics
                        st.markdown("### Word Statistics")
                        st.markdown(f"""
                        - Total Words: {analysis['word_count']}
                        - Unique Words: {analysis['unique_word_count']}
                        - Sentences: {analysis['sentence_count']}
                        """)
                        
                        # Readability Metrics
                        st.markdown("### Readability Metrics")
                        metrics = analysis['readability_metrics']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Avg. Sentence Length",
                                f"{metrics['avg_sentence_length']:.1f} words",
                                help="Average number of words per sentence"
                            )
                        with col2:
                            st.metric(
                                "Avg. Word Length",
                                f"{metrics['avg_word_length']:.1f} chars",
                                help="Average number of characters per word"
                            )
                        with col3:
                            st.metric(
                                "Unique Word Ratio",
                                f"{metrics['unique_word_ratio']:.1%}",
                                help="Ratio of unique words to total words"
                            )
                        
                        # Add readability level assessment
                        readability_level = 'complex' if metrics['avg_sentence_length'] > 20 else 'moderate' if metrics['avg_sentence_length'] > 15 else 'simple'
                        st.markdown(f"""
                        <div style="background-color: #1a1a1a; padding: 15px; border-radius: 5px; color: white; margin-top: 10px;">
                            <strong>Readability Level:</strong> {readability_level}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Named Entities
                        st.markdown("### Named Entities")
                        if analysis['entities']:
                            for entity_type, entities in analysis['entities'].items():
                                st.markdown(f"**{entity_type}:**")
                                for entity in entities[:5]:  # Show top 5 entities per type
                                    st.markdown(f"- {entity}")
                        else:
                            st.info("No named entities found in the text.")
        
        # Abstract Generation with better UI
        st.subheader("📝 Abstract Generation")
        col1, col2 = st.columns([2, 1])
        with col1:
            abstract_length = st.selectbox(
                "Select abstract length",
                ["short", "medium", "long"],
                help="Short: ~100 words (5-7s), Medium: ~200 words (7-12s), Long: ~300 words (12-18s)"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Abstract", key="generate_abstract"):
                with st.spinner("Initializing model..."):
                    try:
                        abstract, key_phrases = generate_abstract(text, abstract_length)
                        if abstract.startswith("Error:"):
                            st.error(abstract)
                        else:
                            st.session_state.abstract = abstract
                            st.session_state.current_abstract = abstract  # Store the current abstract
                            
                            # Display abstract with full width
                            st.markdown("#### Abstract")
                            formatted_abstract = f'''
                            <div style="background-color: #008000; padding: 20px; border-radius: 5px; color: white; line-height: 1.6; width: 100%; margin-bottom: 20px;">
                                {abstract}
                            </div>
                            '''
                            st.markdown(formatted_abstract, unsafe_allow_html=True)
                            
                            # Add download button for abstract
                            st.markdown("#### Download Abstract")
                            abstract_text = download_abstract(abstract)
                            st.download_button(
                                label="Download Abstract as TXT",
                                data=abstract_text,
                                file_name="abstract.txt",
                                mime="text/plain"
                            )
                            
                            # Display translations in a grid layout
                            if st.session_state.translated_text:
                                st.markdown("### Translations")
                                # Calculate number of columns based on number of translations
                                num_translations = len(st.session_state.translated_text)
                                num_cols = min(3, num_translations)  # Maximum 3 columns
                                
                                # Create columns for translations
                                trans_cols = st.columns(num_cols)
                                
                                # Distribute translations across columns
                                for idx, (lang, trans_text) in enumerate(st.session_state.translated_text.items()):
                                    col_idx = idx % num_cols
                                    with trans_cols[col_idx]:
                                        # Get language name
                                        lang_name = {
                                            "es": "Spanish",
                                            "fr": "French",
                                            "de": "German",
                                            "hi": "Hindi",
                                            "zh-cn": "Chinese (Simplified)",
                                            "ja": "Japanese",
                                            "ko": "Korean",
                                            "ru": "Russian",
                                            "ar": "Arabic",
                                            "pt": "Portuguese"
                                        }.get(lang, lang)
                                        
                                        st.markdown(f"#### {lang_name}")
                                        formatted_translation = f'''
                                        <div style="background-color:rgb(75, 75, 75); padding: 20px; border-radius: 5px; color: white; line-height: 1.6; margin-bottom: 10px;">
                                            {trans_text}
                                        </div>
                                        '''
                                        st.markdown(formatted_translation, unsafe_allow_html=True)
                                        
                                        # Add download button for translation
                                        translation_text = download_abstract(trans_text)
                                        st.download_button(
                                            label=f"Download {lang_name} Translation",
                                            data=translation_text,
                                            file_name=f"translation_{lang}.txt",
                                            mime="text/plain"
                                        )
                                        
                                        # Add audio player if available
                                        if lang in st.session_state.audio_files:
                                            st.markdown("#### 🔊 Listen")
                                            st.markdown(f"""
                                            <div style='background-color:rgb(75, 75, 75); padding: 15px; border-radius: 5px;'>
                                                {get_audio_player(st.session_state.audio_files[lang])}
                                            </div>
                                            """, unsafe_allow_html=True)
                            else:
                                st.info("Generate translations to see them here")
                            
                            # Save to history
                            save_to_history(text, abstract, st.session_state.analysis)
                    except Exception as e:
                        st.error(f"An error occurred while generating the abstract: {str(e)}")
        
        # new insights section
        if st.session_state.current_abstract:
            st.markdown("---")
            st.subheader("📊 Text Insights")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Generate Insights", key="generate_insights"):
                    with st.spinner("Generating insights..."):
                        try:
                            # Get text analysis
                            analysis = analyze_text(st.session_state.current_abstract)
                            
                            # Generate insights based on analysis
                            insights = []
                            
                            # Key themes from word frequency
                            top_words = list(analysis['word_frequency'].keys())[:5]
                            insights.append(f"**Key Themes:** {', '.join(top_words)}")
                            
                            # Readability analysis
                            metrics = analysis['readability_metrics']
                            readability_level = 'complex' if metrics['avg_sentence_length'] > 20 else 'moderate' if metrics['avg_sentence_length'] > 15 else 'simple'
                            insights.append(f"**Readability:** The text has an average sentence length of {metrics['avg_sentence_length']:.1f} words, suggesting {readability_level} readability.")
                            
                            # Named entities analysis
                            if analysis['entities']:
                                entity_types = list(analysis['entities'].keys())
                                entity_counts = {k: len(v) for k, v in analysis['entities'].items()}
                                top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                insights.append(f"**Key Entities:** The text contains {sum(entity_counts.values())} named entities across {len(entity_types)} categories. Top categories: {', '.join([f'{k} ({v})' for k, v in top_entities])}.")
                            
                            # Word statistics and vocabulary analysis
                            vocab_level = 'rich' if analysis['readability_metrics']['unique_word_ratio'] > 0.7 else 'moderate' if analysis['readability_metrics']['unique_word_ratio'] > 0.5 else 'basic'
                            insights.append(f"**Vocabulary:** The text uses {analysis['unique_word_count']} unique words out of {analysis['word_count']} total words, indicating {vocab_level} vocabulary.")
                            
                            # Part of speech analysis
                            pos_dist = analysis['pos_distribution']
                            if 'NOUN' in pos_dist and 'VERB' in pos_dist:
                                noun_verb_ratio = pos_dist['NOUN'] / pos_dist['VERB']
                                style = 'descriptive' if noun_verb_ratio > 1.5 else 'balanced' if noun_verb_ratio > 0.8 else 'action-oriented'
                                insights.append(f"**Writing Style:** The text has a {style} style with {pos_dist['NOUN']} nouns and {pos_dist['VERB']} verbs.")
                            
                            # Key phrases analysis
                            if analysis['noun_phrases']:
                                top_phrases = Counter(analysis['noun_phrases']).most_common(3)
                                insights.append(f"**Key Phrases:** The most prominent phrases are: {', '.join([f'{phrase} ({count})' for phrase, count in top_phrases])}.")
                            
                            # Format and display insights
                            formatted_insights = '<br>'.join(insights)
                            st.session_state.insights = formatted_insights
                            
                            st.markdown("#### Analysis")
                            st.markdown(f'''
                            <div style="background-color: #1a1a1a; padding: 20px; border-radius: 5px; color: #ffffff; line-height: 1.6; margin-bottom: 20px;">
                                {formatted_insights}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Add 3D visualizations to insights
                            st.markdown("#### 3D Visualizations")
                            
                            # 3D Word Frequency Analysis
                            st.markdown("##### Word Frequency Analysis")
                            words = list(analysis['word_frequency'].keys())
                            frequencies = list(analysis['word_frequency'].values())
                            word_lengths = [len(word) for word in words]
                            
                            fig_3d = px.scatter_3d(
                                x=words,
                                y=frequencies,
                                z=word_lengths,
                                title="3D Word Analysis: Frequency vs Word Length",
                                labels={'x': 'Words', 'y': 'Frequency', 'z': 'Word Length'},
                                color=frequencies,
                                color_continuous_scale='Viridis',
                                size=frequencies,
                                size_max=20,
                                text=words,  # Add word labels
                                hover_name=words  # Use words as hover labels
                            )
                            fig_3d.update_layout(
                                scene=dict(
                                    camera=dict(
                                        up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=0),
                                        eye=dict(x=1.5, y=1.5, z=1.5)
                                    )
                                )
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                            
                            
                            
                            # Add download button for insights
                            insights_text = download_abstract('\n'.join(insights))
                            st.download_button(
                                label="Download Insights as TXT",
                                data=insights_text,
                                file_name="text_insights.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
        else:
            st.info("Generate an abstract first to enable insights generation")
        
        
        # Translation with improved UI
        if st.session_state.abstract:
            st.subheader("🌐 Translation")
            col1, col2 = st.columns([2, 1])
            with col1:
                target_lang = st.selectbox(
                    "Select target language",
                    ["es", "fr", "de", "hi", "zh-cn", "ja", "ko", "ru", "ar", "pt"],
                    format_func=lambda x: {
                        "es": "Spanish",
                        "fr": "French",
                        "de": "German",
                        "hi": "Hindi",
                        "zh-cn": "Chinese (Simplified)",
                        "ja": "Japanese",
                        "ko": "Korean",
                        "ru": "Russian",
                        "ar": "Arabic",
                        "pt": "Portuguese"
                    }[x]
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Translate", key="translate"):
                    with st.spinner("Translating..."):
                        try:
                            translated_text = translate_text(st.session_state.abstract, target_lang)
                            if not translated_text.startswith("Error:"):
                                st.session_state.translated_text[target_lang] = translated_text
                                
                                # Display translation in a nice card
                                st.markdown("### Translation")
                                # Split translation into paragraphs and display each
                                translation_paragraphs = translated_text.split('. ')
                                formatted_translation = f'''
                                <div style="background-color:rgb(75, 75, 75); padding: 15px; border-radius: 5px;">
                                    {translated_text}
                                </div>
                                '''
                                st.markdown(formatted_translation, unsafe_allow_html=True)
                                
                                # Add download button for translation
                                st.markdown("### Download Translation")
                                translation_text = download_abstract(translated_text)
                                st.download_button(
                                    label=f"Download Translation as TXT",
                                    data=translation_text,
                                    file_name=f"translation_{target_lang}.txt",
                                    mime="text/plain"
                                )
                                
                                # Add text-to-speech functionality
                                if target_lang not in st.session_state.audio_files:
                                    temp_dir = tempfile.gettempdir()
                                    audio_file = os.path.join(temp_dir, f"translation_{target_lang}.mp3")
                                    text_to_speech(translated_text, target_lang, audio_file)
                                    st.session_state.audio_files[target_lang] = audio_file
                                
                                # Display audio player with improved styling
                                st.markdown("### 🔊 Listen to Translation")
                                st.markdown(f"""
                                <div style='background-color:rgb(75, 75, 75); padding: 15px; border-radius: 5px;'>
                                    {get_audio_player(st.session_state.audio_files[target_lang])}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"An error occurred during translation: {str(e)}")
                            st.error("Please try again or select a different language.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try uploading a different file or check if the file is corrupted.")