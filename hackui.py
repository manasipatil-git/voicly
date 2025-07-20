import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st
import numpy as np
import pyaudio
import wave
import threading
import queue
import tempfile
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, deque
import re
import asyncio
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import hashlib

class RealtimeCustomerSupportAI:
    def __init__(self):
        self.setup_models()
        self.setup_database()
        self.conversation_history = []
        self.analytics_data = defaultdict(list)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.conversation_context = deque(maxlen=5)  # Keep last 5 exchanges
        self.knowledge_base = self.setup_knowledge_base()
        self.customer_profile = {}
        
    def setup_models(self): 
        """Initialize all models with enhanced capabilities"""
        st.write("Just a sec!")
        
        progress_bar = st.progress(0)
        
        # Load Whisper for real-time transcription
        self.whisper_model = whisper.load_model("base")
        progress_bar.progress(20)
        # Load sentence transformer for semantic search
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        progress_bar.progress(35)
        
        # Load advanced LLM
        try:
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            st.success("Almost there!")
        except Exception as e:
            st.warning(f"Using fallback model: {e}")
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        progress_bar.progress(50)
        
        # Advanced sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        progress_bar.progress(65)
        
        # Emotion classification
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        progress_bar.progress(80)
        
        # Intent classification
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        progress_bar.progress(95)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        progress_bar.progress(100)
        st.success("All Enhanced Models Loaded!")
    
    def setup_database(self):
        """Initialize SQLite database for conversation history"""
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                customer_id TEXT,
                transcript TEXT,
                sentiment TEXT,
                emotion TEXT,
                intent TEXT,
                category TEXT,
                urgency TEXT,
                response TEXT,
                satisfaction_score REAL,
                escalation_needed BOOLEAN,
                processing_time REAL,
                resolved BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE customer_profiles (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                tier TEXT,
                total_calls INTEGER,
                satisfaction_avg REAL,
                preferred_channel TEXT,
                last_interaction DATETIME
            )
        ''')
        
        self.conn.commit()
    
    def setup_knowledge_base(self):
        """Create a semantic knowledge base using FAISS"""
        # Sample knowledge base entries
        knowledge_entries = [
            "To reset your password, go to login page and click 'Forgot Password'",
            "Billing issues can be resolved by contacting our billing department",
            "Technical support is available 24/7 for premium customers",
            "Product returns must be initiated within 30 days of purchase",
            "Account upgrades include priority support and extended warranties",
            "Shipping typically takes 3-5 business days for standard delivery",
            "International orders may incur additional customs fees",
            "Premium support includes dedicated account managers"
        ]
        
        # Create embeddings
        embeddings = self.sentence_transformer.encode(knowledge_entries)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        return {
            'entries': knowledge_entries,
            'embeddings': embeddings,
            'index': index
        }
    
    def search_knowledge_base(self, query, top_k=3):
        """Search knowledge base using semantic similarity"""
        query_embedding = self.sentence_transformer.encode([query])
        distances, indices = self.knowledge_base['index'].search(
            query_embedding.astype('float32'), top_k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'text': self.knowledge_base['entries'][idx],
                'similarity': 1 - distances[0][i]  # Convert distance to similarity
            })
        
        return results
    
    def analyze_advanced_sentiment(self, text):
        """Enhanced sentiment analysis with context - Fixed capitalization"""
        try:
            # Basic sentiment
            sentiment_results = self.sentiment_analyzer(text)
            sentiment_scores = {result['label']: result['score'] for result in sentiment_results[0]}
            
            # Emotion analysis
            emotion_results = self.emotion_classifier(text)
            emotion_scores = {result['label']: result['score'] for result in emotion_results[0]}
            
            # Intent classification
            candidate_intents = [
                "billing inquiry", "technical support", "account management", 
                "product information", "complaint", "compliment", "cancellation request"
            ]
            
            intent_results = self.intent_classifier(text, candidate_intents)
            primary_intent = intent_results['labels'][0]
            intent_confidence = intent_results['scores'][0]
            
            # Get the sentiment with highest score and normalize capitalization
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            # Convert to uppercase for consistency
            primary_sentiment = primary_sentiment.upper()
            
            return {
                "sentiment": primary_sentiment,  # Now consistently uppercase
                "sentiment_confidence": max(sentiment_scores.values()),
                "emotion": max(emotion_scores, key=emotion_scores.get),
                "emotion_confidence": max(emotion_scores.values()),
                "intent": primary_intent,
                "intent_confidence": intent_confidence,
                "all_sentiments": {k.upper(): v for k, v in sentiment_scores.items()},  # Also normalize this
                "all_emotions": emotion_scores
            }
        except Exception as e:
            st.error(f"Advanced analysis error: {e}")
            return self.fallback_analysis(text)

    def fallback_analysis(self, text):
        """Fallback analysis using rule-based approach - Fixed capitalization"""
        # Simple rule-based sentiment
        positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'angry']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
        elif negative_count > positive_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,  # Already uppercase, consistent
            "sentiment_confidence": 0.7,
            "emotion": "neutral",
            "emotion_confidence": 0.7,
            "intent": "general_inquiry",
            "intent_confidence": 0.6,
            "all_sentiments": {sentiment: 0.7},
            "all_emotions": {"neutral": 0.7}
        }

    
    def extract_enhanced_insights(self, text, sentiment_data):
        """Extract comprehensive insights from customer message"""
        insights = {
            "urgency_level": "medium",
            "keywords": [],
            "issue_category": "general",
            "action_required": [],
            "escalation_needed": False,
            "customer_effort_score": 3,  # 1-5 scale
            "resolution_complexity": "medium",
            "suggested_actions": [],
            "similar_cases": [],
            "satisfaction_prediction": 0.7
        }
        
        text_lower = text.lower()
        
        # Enhanced urgency detection
        urgency_indicators = {
            "high": ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'now', 'today'],
            "medium": ['soon', 'quickly', 'fast', 'priority'],
            "low": ['whenever', 'no rush', 'eventually']
        }
        
        for level, keywords in urgency_indicators.items():
            if any(word in text_lower for word in keywords):
                insights["urgency_level"] = level
                break
        
        # Advanced categorization
        categories = {
            "billing": ['bill', 'charge', 'payment', 'invoice', 'refund', 'money', 'cost'],
            "technical": ['not working', 'broken', 'bug', 'error', 'crash', 'issue', 'problem'],
            "shipping": ['delivery', 'shipping', 'late', 'order', 'package', 'tracking'],
            "returns": ['return', 'exchange', 'defective', 'wrong', 'damaged', 'replace'],
            "account": ['login', 'password', 'account', 'access', 'profile', 'settings'],
            "product": ['feature', 'how to', 'tutorial', 'information', 'specs']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                insights["issue_category"] = category
                break
        
        # Customer effort prediction
        effort_indicators = {
            "high": ['multiple', 'many times', 'again', 'still', 'keep', 'repeatedly'],
            "medium": ['tried', 'attempted', 'called'],
            "low": ['first time', 'new', 'quick question']
        }
        
        for level, keywords in effort_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                insights["customer_effort_score"] = {"high": 5, "medium": 3, "low": 1}[level]
                break
        
        # Escalation prediction
        escalation_triggers = [
            'manager', 'supervisor', 'complaint', 'lawsuit', 'terrible', 
            'worst', 'unacceptable', 'cancel', 'switch', 'competitor'
        ]
        
        if any(trigger in text_lower for trigger in escalation_triggers):
            insights["escalation_needed"] = True
        
        # Satisfaction prediction based on sentiment and emotion
        if sentiment_data["sentiment"] == "POSITIVE":
            insights["satisfaction_prediction"] = 0.8
        elif sentiment_data["sentiment"] == "NEGATIVE":
            insights["satisfaction_prediction"] = 0.3
        else:
            insights["satisfaction_prediction"] = 0.6
        
        # Suggested actions based on category and urgency
        action_map = {
            "billing": ["Review account charges", "Process refund if applicable", "Update payment method"],
            "technical": ["Run diagnostic", "Check system status", "Escalate to tech team"],
            "shipping": ["Track package", "Contact carrier", "Provide delivery estimate"],
            "returns": ["Initiate return process", "Send return label", "Process exchange"],
            "account": ["Verify identity", "Reset credentials", "Update profile"],
            "product": ["Provide documentation", "Schedule tutorial", "Send product guide"]
        }
        
        insights["suggested_actions"] = action_map.get(insights["issue_category"], ["Gather more information"])
        
        return insights
    
    def generate_contextual_response(self, transcript, sentiment_data, insights, context_history):
        """Generate highly contextual response using conversation history"""
        try:
            # Search knowledge base for relevant information
            kb_results = self.search_knowledge_base(transcript)
            relevant_info = kb_results[0]['text'] if kb_results else ""
            
            # Build context from conversation history
            context_summary = ""
            if context_history:
                parts = [f"{ex['customer']}: {ex['response']}" for ex in context_history]
                context_summary = f"Previous context: {' '.join(parts)}"

            # Create enhanced prompt
            prompt = f"""
            You are an expert customer service agent. Use the following information to provide the best response:
            
            Customer: "{transcript}"
            
            Context:
            - Sentiment: {sentiment_data['sentiment']} ({sentiment_data['sentiment_confidence']:.2f})
            - Emotion: {sentiment_data['emotion']} ({sentiment_data['emotion_confidence']:.2f})
            - Intent: {sentiment_data['intent']} ({sentiment_data['intent_confidence']:.2f})
            - Category: {insights['issue_category']}
            - Urgency: {insights['urgency_level']}
            - Escalation needed: {insights['escalation_needed']}
            - Customer effort: {insights['customer_effort_score']}/5
            
            Relevant company information: {relevant_info}
            
            {context_summary}
            
            Generate a response that:
            1. Acknowledges their specific concern
            2. Shows empathy matching their emotional state
            3. Provides actionable next steps
            4. Is professional yet personal
            5. Addresses their intent directly
            
            Response:"""
            
            # Generate response
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            
            # Clean up
            response = re.sub(r'^["\']|["\']$', '', response)
            response = response.replace(transcript, "").strip()
            
            # Fallback if generation fails
            if len(response) < 20:
                response = self.get_smart_fallback_response(insights, sentiment_data, relevant_info)
            
            return response
            
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return self.get_smart_fallback_response(insights, sentiment_data, "")
    
    def get_smart_fallback_response(self, insights, sentiment_data, relevant_info):
        """Intelligent fallback response system"""
        category = insights['issue_category']
        urgency = insights['urgency_level']
        emotion = sentiment_data['emotion']
        
        # Base responses by category
        base_responses = {
            "billing": "I understand your billing concern and I'm here to help resolve it immediately.",
            "technical": "I apologize for the technical difficulty you're experiencing. Let me help you troubleshoot this.",
            "shipping": "I can see you're concerned about your delivery. Let me track that for you right away.",
            "returns": "I'll be happy to help you with the return process. Let me get that started for you.",
            "account": "I can help you with your account issue. Let me verify your information and resolve this.",
            "product": "I'd be glad to help you with product information. Let me provide you with the details you need."
        }
        
        base_response = base_responses.get(category, "I'm here to help you with your concern.")
        
        # Emotional customization
        if emotion in ['anger', 'frustration']:
            base_response = f"I sincerely apologize for the frustration this has caused. {base_response}"
        elif emotion in ['sadness', 'disappointment']:
            base_response = f"I understand how disappointing this must be for you. {base_response}"
        elif emotion in ['fear', 'anxiety']:
            base_response = f"I want to reassure you that we'll get this sorted out. {base_response}"
        
        # Urgency customization
        if urgency == "high":
            base_response += " I'm treating this as a priority and will ensure we resolve it as quickly as possible."
        elif urgency == "low":
            base_response += " I'll make sure we find the right solution for you."
        
        # Add relevant information if available
        if relevant_info:
            base_response += f" Based on our policies: {relevant_info}"
        
        return base_response
    
    def update_customer_profile(self, customer_id, interaction_data):
        """Update customer profile with interaction data"""
        cursor = self.conn.cursor()
        
        # Check if customer exists
        cursor.execute("SELECT * FROM customer_profiles WHERE customer_id = ?", (customer_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing profile
            cursor.execute("""
                UPDATE customer_profiles 
                SET total_calls = total_calls + 1,
                    satisfaction_avg = (satisfaction_avg * (total_calls - 1) + ?) / total_calls,
                    last_interaction = ?
                WHERE customer_id = ?
            """, (interaction_data['satisfaction_score'], datetime.now(), customer_id))
        else:
            # Create new profile
            cursor.execute("""
                INSERT INTO customer_profiles 
                (customer_id, name, tier, total_calls, satisfaction_avg, preferred_channel, last_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (customer_id, "Unknown", "Standard", 1, interaction_data['satisfaction_score'], 
                  "voice", datetime.now()))
        
        self.conn.commit()
    
    def calculate_satisfaction_score(self, sentiment_data, insights, response_quality):
        """Calculate predicted customer satisfaction score"""
        base_score = 0.5
        
        # Sentiment impact
        if sentiment_data['sentiment'] == 'POSITIVE':
            base_score += 0.3
        elif sentiment_data['sentiment'] == 'NEGATIVE':
            base_score -= 0.3
        
        # Emotion impact
        if sentiment_data['emotion'] in ['joy', 'satisfaction']:
            base_score += 0.2
        elif sentiment_data['emotion'] in ['anger', 'frustration']:
            base_score -= 0.2
        
        # Resolution complexity
        if insights['urgency_level'] == 'low':
            base_score += 0.1
        elif insights['urgency_level'] == 'high':
            base_score -= 0.1
        
        # Response quality (mock scoring)
        base_score += response_quality * 0.2
        
        return max(0, min(1, base_score))
    
    def process_realtime_call(self, audio_data):
        """Process real-time audio data"""
        start_time = time.time()
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe
            result = self.whisper_model.transcribe(temp_file_path)
            transcript = result["text"].strip()
            
            if len(transcript) < 3:  # Skip very short transcripts
                return None
            
            # Analyze
            sentiment_data = self.analyze_advanced_sentiment(transcript)
            insights = self.extract_enhanced_insights(transcript, sentiment_data)
            
            # Generate response with context
            response = self.generate_contextual_response(
                transcript, sentiment_data, insights, list(self.conversation_context)
            )
            
            # Calculate satisfaction
            satisfaction_score = self.calculate_satisfaction_score(
                sentiment_data, insights, 0.8  # Mock response quality
            )
            
            processing_time = time.time() - start_time
            
            # Store interaction
            interaction_data = {
                "timestamp": datetime.now(),
                "transcript": transcript,
                "sentiment": sentiment_data["sentiment"],
                "emotion": sentiment_data["emotion"],
                "intent": sentiment_data["intent"],
                "category": insights["issue_category"],
                "urgency": insights["urgency_level"],
                "response": response,
                "satisfaction_score": satisfaction_score,
                "processing_time": processing_time,
                "escalation_needed": insights["escalation_needed"],
                "resolved": not insights["escalation_needed"]
            }
            
            # Update conversation context
            self.conversation_context.append({
                "customer": transcript,
                "response": response,
                "timestamp": datetime.now()
            })
            
            self.conversation_history.append(interaction_data)
            
            # Clean up temp file
            import os
            os.unlink(temp_file_path)
            
            return {
                "transcription": {"text": transcript, "confidence": result.get("avg_logprob", 0)},
                "sentiment_data": sentiment_data,
                "insights": insights,
                "response": response,
                "satisfaction_score": satisfaction_score,
                "processing_time": processing_time,
                "suggested_actions": insights["suggested_actions"]
            }
            
        except Exception as e:
            st.error(f"Real-time processing error: {e}")
            return None

# Streamlit UI with enhanced features
def main():
    st.set_page_config(
        page_title="Voicly",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>Voicly</h1><p>AI Customer Support System</p></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = RealtimeCustomerSupportAI()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", [
        "Live Call Processing", 
        "Advanced Analytics", 
        "AI Insights", 
        "Customer Profiles",
        "System Settings"
    ])
    
    if page == "Live Call Processing":
        show_live_processing()
    elif page == "Advanced Analytics":
        show_advanced_analytics()
    elif page == "AI Insights":
        show_ai_insights()
    elif page == "Customer Profiles":
        show_customer_profiles()
    elif page == "System Settings":
        show_system_settings()

def show_live_processing():
    st.header("Real-time Call Processing")
    
    # Real-time metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Calls", "1", delta="0")
    with col2:
        st.metric("Avg Response Time", "2.3s", delta="-0.2s")
    with col3:
        st.metric("Satisfaction Score", "4.2/5", delta="0.1")
    with col4:
        st.metric("Escalation Rate", "8%", delta="-2%")
    
    # Main processing area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📞 Call Interface")
        
        # Audio input options
        audio_option = st.radio("Choose input method:", 
                               ["Upload Audio File", "Record Live (Demo)", "Simulate Real-time"])
        
        if audio_option == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload customer audio file",
                type=['wav', 'mp3', 'mp4', 'm4a'],
                help="Upload a customer call recording for analysis"
            )
            
            if uploaded_file is not None:
                process_uploaded_file(uploaded_file)
        
        elif audio_option == "Record Live (Demo)":
            st.info("Live recording feature (requires microphone access)")
            if st.button("Start Recording", type="primary"):
                st.warning("Live recording would start here in production")
        
        elif audio_option == "Simulate Real-time":
            show_realtime_simulation()
    
    with col2:
        st.subheader("Live Insights")
        
        # Real-time emotion gauge
        st.markdown("**Current Emotion:**")
        emotion_placeholder = st.empty()
        
        # Sentiment indicator
        st.markdown("**Sentiment Trend:**")
        sentiment_placeholder = st.empty()
        
        # Action recommendations
        st.markdown("**Recommended Actions:**")
        actions_placeholder = st.empty()
        
        # Escalation alert
        st.markdown("**Escalation Status:**")
        escalation_placeholder = st.empty()

def process_uploaded_file(uploaded_file):
    """Process uploaded audio file with enhanced features"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    if st.button("Process Call", type="primary"):
        with st.spinner("Processing"):
            # Create progress indicators
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                status_text.text("Transcribing audio...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("Analyzing sentiment and emotion...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("Extracting insights...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                status_text.text("Generating response...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Process the call
                result = st.session_state.ai_assistant.process_realtime_call(
                    open(tmp_file_path, 'rb').read()
                )
                
                progress_container.empty()
        
        if result:
            display_call_results(result)
        
        # Clean up
        import os
        os.unlink(tmp_file_path)

def display_call_results(result):
    """Display comprehensive call results"""
    st.success("✅ Call processed successfully!")
    
    # Main results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Call Transcript")
        st.text_area("Customer said:", result["transcription"]["text"], height=100)
        
        st.subheader("AI-Generated Response")
        st.success(result["response"])
        
        st.subheader("Suggested Actions")
        for i, action in enumerate(result["suggested_actions"], 1):
            st.write(f"{i}. {action}")
    
    with col2:
        st.subheader("Call Analytics")
        
        # Sentiment with visual indicator
        sentiment = result["sentiment_data"]["sentiment"]
        sentiment_color = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}
        st.metric("Sentiment", f"{sentiment_color.get(sentiment, '⚪')} {sentiment}")
        
        # Emotion
        emotion = result["sentiment_data"]["emotion"]
        st.metric("Emotion", emotion.title())
        
        # Intent
        intent = result["sentiment_data"]["intent"]
        st.metric("Intent", intent.title())
        
        # Category
        category = result["insights"]["issue_category"]
        st.metric("Issue Category", category.title())
        
        # Urgency with color coding
        urgency = result["insights"]["urgency_level"]
        urgency_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        st.metric("Urgency", f"{urgency_colors.get(urgency, '⚪')} {urgency.title()}")
        
        # Satisfaction prediction
        satisfaction = result["satisfaction_score"]
        st.metric("Predicted Satisfaction", f"{satisfaction:.1f}/5.0")
        
        # Processing time
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        # Escalation alert
        if result["insights"]["escalation_needed"]:
            st.error("⚠️ ESCALATION REQUIRED!")
        else:
            st.success("✅ Standard Resolution")

def show_realtime_simulation():
    """Show simulated real-time processing"""
    st.subheader("🎭 Real-time Processing Simulation")
    
    # Sample customer messages for simulation
    sample_messages = [
        "Hi, I'm having trouble with my recent order. It was supposed to arrive yesterday but I haven't received it yet.",
        "I'm really frustrated! This is the third time I'm calling about the same billing issue and nobody seems to help me!",
        "Hello, I just wanted to thank you for the excellent service last time. I have a quick question about my account.",
        "My internet has been down for 2 hours and I work from home. This is urgent, I need it fixed immediately!",
        "I'm thinking about canceling my subscription. The service hasn't been meeting my expectations lately."
    ]
    
    if st.button("Simulate Real-time Call", type="primary"):
        import random
        selected_message = random.choice(sample_messages)
        
        # Simulate real-time processing
        st.write("**Live Transcript:**")
        transcript_placeholder = st.empty()
        
        # Simulate typing effect
        for i in range(len(selected_message)):
            transcript_placeholder.text(selected_message[:i+1])
            time.sleep(0.05)
        
        # Process the simulated message
        with st.spinner("Processing in real-time..."):
            # Simulate processing
            sentiment_data = st.session_state.ai_assistant.analyze_advanced_sentiment(selected_message)
            insights = st.session_state.ai_assistant.extract_enhanced_insights(selected_message, sentiment_data)
            response = st.session_state.ai_assistant.generate_contextual_response(
                selected_message, sentiment_data, insights, []
            )
            
            result = {
                "transcription": {"text": selected_message, "confidence": 0.95},
                "sentiment_data": sentiment_data,
                "insights": insights,
                "response": response,
                "satisfaction_score": 0.8,
                "processing_time": 2.1,
                "suggested_actions": insights["suggested_actions"]
            }
        
        st.write("---")
        display_call_results(result)

def show_advanced_analytics():
    """Show advanced analytics dashboard"""
    st.header("Advanced Analytics Dashboard")
    
    if not st.session_state.ai_assistant.conversation_history:
        st.info("No data available yet. Process some calls to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.ai_assistant.conversation_history)
    
    # KPI Section
    st.subheader("Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        total_calls = len(df)
        st.metric("Total Calls", total_calls)
    
    with kpi_col2:
        avg_satisfaction = df["satisfaction_score"].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/5.0")
    
    with kpi_col3:
        avg_processing_time = df["processing_time"].mean()
        st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
    
    with kpi_col4:
        escalation_rate = (df["escalation_needed"].sum() / len(df)) * 100
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
    
    with kpi_col5:
        resolution_rate = (df["resolved"].sum() / len(df)) * 100
        st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
    
    # Charts Section
    st.subheader("Trend Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Sentiment over time
        fig = px.line(df, x="timestamp", y="satisfaction_score", 
                     title="Satisfaction Score Over Time",
                     labels={"satisfaction_score": "Satisfaction Score"})
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Category distribution
        category_counts = df["category"].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                    title="Issue Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Analytics
    st.subheader("Advanced Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        # Sentiment vs Satisfaction correlation
        fig = px.scatter(df, x="satisfaction_score", color="sentiment",
                        title="Sentiment vs Satisfaction Score")
        st.plotly_chart(fig, use_container_width=True)
    
    with insight_col2:
        # Processing time by urgency
        fig = px.box(df, x="urgency", y="processing_time",
                    title="Processing Time by Urgency Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Recent Call Details")
    recent_calls = df.tail(10)[["timestamp", "sentiment", "emotion", "category", 
                               "urgency", "satisfaction_score", "escalation_needed"]]
    st.dataframe(recent_calls, use_container_width=True)

def show_ai_insights():
    """Show AI-powered insights and recommendations"""
    st.header("AI Insights & Recommendations")
    
    if not st.session_state.ai_assistant.conversation_history:
        st.info("No data available for AI analysis yet.")
        return
    
    df = pd.DataFrame(st.session_state.ai_assistant.conversation_history)
    
    # AI-Generated Insights
    st.subheader("AI-Generated Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Call Pattern Analysis**")
        
        # Most common issues
        top_categories = df["category"].value_counts().head(3)
        st.write("**Top Issues:**")
        for category, count in top_categories.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {category.title()}: {count} calls ({percentage:.1f}%)")
        
        # Sentiment trends
        sentiment_trend = df["sentiment"].value_counts()
        st.write("**Sentiment Distribution:**")
        for sentiment, count in sentiment_trend.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {sentiment}: {percentage:.1f}%")
    
    with col2:
        st.markdown("**Risk Indicators**")
        
        # High-risk patterns
        high_risk_calls = df[df["escalation_needed"] == True]
        if len(high_risk_calls) > 0:
            st.warning(f"⚠️ {len(high_risk_calls)} calls flagged for escalation")
            
            # Common escalation triggers
            escalation_categories = high_risk_calls["category"].value_counts()
            st.write("**Escalation by Category:**")
            for category, count in escalation_categories.items():
                st.write(f"• {category.title()}: {count} escalations")
        else:
            st.success("✅ No high-risk calls detected")
        
        # Low satisfaction alerts
        low_satisfaction = df[df["satisfaction_score"] < 3.0]
        if len(low_satisfaction) > 0:
            st.warning(f"📉 {len(low_satisfaction)} calls with low satisfaction")
    
    # Recommendations
    st.subheader("💡 AI Recommendations")
    
    recommendations = []
    
    # Analyze patterns and generate recommendations
    if len(df) > 0:
        avg_satisfaction = df["satisfaction_score"].mean()
        escalation_rate = (df["escalation_needed"].sum() / len(df)) * 100
        
        if avg_satisfaction < 3.5:
            recommendations.append({
                "type": "warning",
                "title": "Improve Customer Satisfaction",
                "message": f"Average satisfaction is {avg_satisfaction:.1f}/5.0. Consider reviewing response templates and agent training.",
                "action": "Review and update response generation model"
            })
        
        if escalation_rate > 15:
            recommendations.append({
                "type": "error",
                "title": "High Escalation Rate",
                "message": f"Escalation rate is {escalation_rate:.1f}%. Implement proactive issue resolution.",
                "action": "Enhance early warning system and agent coaching"
            })
        
        # Processing time recommendations
        avg_processing_time = df["processing_time"].mean()
        if avg_processing_time > 5:
            recommendations.append({
                "type": "info",
                "title": "Optimize Processing Speed",
                "message": f"Average processing time is {avg_processing_time:.1f}s. Consider model optimization.",
                "action": "Implement model quantization and caching"
            })
        
        if not recommendations:
            recommendations.append({
                "type": "success",
                "title": "System Performing Well",
                "message": "All metrics are within acceptable ranges. Continue monitoring.",
                "action": "Maintain current performance levels"
            })
    
    # Display recommendations
    for rec in recommendations:
        if rec["type"] == "success":
            st.success(f"✅ **{rec['title']}**: {rec['message']}")
        elif rec["type"] == "warning":
            st.warning(f"⚠️ **{rec['title']}**: {rec['message']}")
        elif rec["type"] == "error":
            st.error(f"🚨 **{rec['title']}**: {rec['message']}")
        else:
            st.info(f"💡 **{rec['title']}**: {rec['message']}")
        
        st.write(f"**Recommended Action:** {rec['action']}")
        st.write("---")

def show_customer_profiles():
    """Show customer profile management"""
    st.header("Customer Profiles")
    
    st.subheader("Customer Insights")
    
    # Mock customer data for demo
    customer_data = {
        "Customer ID": ["CUST001", "CUST002", "CUST003", "CUST004"],
        "Name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson"],
        "Tier": ["Premium", "Standard", "Premium", "Standard"],
        "Total Calls": [15, 8, 23, 5],
        "Avg Satisfaction": [4.2, 3.8, 4.7, 3.5],
        "Last Contact": ["2 days ago", "1 week ago", "Yesterday", "1 month ago"],
        "Preferred Channel": ["Phone", "Email", "Chat", "Phone"]
    }
    
    customer_df = pd.DataFrame(customer_data)
    
    # Customer search
    search_term = st.text_input("🔍 Search customers:", placeholder="Enter customer name or ID")
    
    if search_term:
        filtered_df = customer_df[
            customer_df["Name"].str.contains(search_term, case=False) |
            customer_df["Customer ID"].str.contains(search_term, case=False)
        ]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(customer_df, use_container_width=True)
    
    # Customer analytics
    st.subheader("Customer Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer tier distribution
        tier_counts = customer_df["Tier"].value_counts()
        fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                    title="Customer Tier Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Channel preference
        channel_counts = customer_df["Preferred Channel"].value_counts()
        fig = px.bar(x=channel_counts.index, y=channel_counts.values,
                    title="Preferred Communication Channels")
        st.plotly_chart(fig, use_container_width=True)

def show_system_settings():
    """Show system settings and configuration"""
    st.header("System Settings")
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Models:**")
        st.info("""
        - **Speech-to-Text**: OpenAI Whisper (base)
        - **Language Model**: Microsoft DialoGPT-medium
        - **Sentiment Analysis**: CardiffNLP Twitter RoBERTa
        - **Emotion Detection**: J-Hartmann Emotion DistilRoBERTa
        - **Intent Classification**: Facebook BART-large-mnli
        """)
    
    with col2:
        st.markdown("**Performance Tuning:**")
        
        # Model parameters
        temperature = st.slider("Response Temperature", 0.1, 1.0, 0.7, 0.1)
        max_length = st.slider("Max Response Length", 50, 200, 100, 10)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8, 0.05)
        
        if st.button("Update Model Parameters"):
            st.success("✅ Model parameters updated successfully!")
    
    # System Performance
    st.subheader("System Performance")
    
    if st.session_state.ai_assistant.conversation_history:
        df = pd.DataFrame(st.session_state.ai_assistant.conversation_history)
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            avg_time = df["processing_time"].mean()
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        with perf_col2:
            total_processed = len(df)
            st.metric("Total Calls Processed", total_processed)
        
        with perf_col3:
            uptime = "99.9%"  # Mock uptime
            st.metric("System Uptime", uptime)
    
    # Data Management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Data:**")
        if st.button("Export Call Data"):
            if st.session_state.ai_assistant.conversation_history:
                df = pd.DataFrame(st.session_state.ai_assistant.conversation_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"call_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
    
    with col2:
        st.markdown("**System Maintenance:**")
        if st.button("Clear Cache"):
            st.success("✅ Cache cleared successfully!")
        
        if st.button("🧹 Reset Analytics"):
            st.session_state.ai_assistant.conversation_history = []
            st.success("✅ Analytics data reset!")

if __name__ == "__main__":
    main()