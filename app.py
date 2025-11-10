import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import speech_recognition as sr
from PIL import Image
import base64
import io
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Get API key from secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = None

# Configure Google AI Client
@st.cache_resource
def configure_google_ai(api_key):
    genai.configure(api_key=api_key)

# Initialize LangChain agent
@st.cache_resource
def get_langchain_agent(api_key):
    """Create LangChain agent with tools"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # <-- FIXED
        temperature=0.7,
        google_api_key=api_key,
        convert_system_message_to_human=True 
    )
    
    # Define tools
    def calculator(expression: str) -> str:
        """Useful for mathematical calculations"""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def text_analyzer(text: str) -> str:
        """Analyze text for sentiment, length, and word count"""
        words = len(text.split())
        chars = len(text)
        return f"Analysis: {words} words, {chars} characters"
    
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for math calculations. Input should be a valid Python expression."
        ),
        Tool(
            name="TextAnalyzer",
            func=text_analyzer,
            description="Analyze text for basic statistics like word count and character count."
        )
    ]
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return agent

# Voice processing functions (No change needed)
def speech_to_text(audio_file_path):
    """Convert audio to text using speech recognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text, None
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Error with speech recognition service: {e}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Image processing functions
def process_image_with_gemini(image_bytes, query, api_key):
    """Process image with Gemini"""
    try:
        configure_google_ai(api_key) 
        
        # <-- FIXED: Use gemini-1.0-pro-vision
        model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
        
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt_parts = [
            query or "Describe this image in detail. What do you see?",
            img
        ]
        
        response = model.generate_content(prompt_parts)
        return response.text, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Text processing with streaming
def get_streaming_response(user_query, chat_history, api_key):
    """Get streaming response from LLM"""
    template = """
    You are a helpful AI assistant. Answer the following question considering the chat history.
    
    Chat history: {chat_history}
    User question: {user_question}
    
    Provide a clear, informative, and friendly response.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # <-- FIXED
        temperature=0.7,
        google_api_key=api_key,
        streaming=True
    )
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query
    })

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">KarmiQ Multimodal AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Interact with AI using Text, Voice, or Images")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not GOOGLE_API_KEY:
            api_key = st.text_input("Google AI Studio API Key", type="password")
            if not api_key:
                st.warning("⚠️ Please enter your Google AI Studio API key to continue")
                st.info("Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
                st.stop()
        else:
            api_key = GOOGLE_API_KEY
            st.success("✅ Google API Key loaded from secrets")
        
        configure_google_ai(api_key)
        
        st.markdown("---")
        
        # Mode selection
        st.subheader("📝 Select Input Mode")
        mode = st.radio(
            "Choose how to interact:",
            ["💬 Text Chat", "🎤 Voice Input", "🖼️ Image Analysis", "🌐 Multimodal"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings
        st.subheader("🎛️ Settings")
        use_agent = st.checkbox("Enable Agent Tools", value=False, 
                               help="Activate tools like calculator and text analysis")
        enable_streaming = st.checkbox("Enable Streaming", value=True,
                                      help="Stream responses in real-time")
        
        st.markdown("---")
        
        # Clear history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.agent_memory.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📚 Instructions"):
            st.markdown("""
            **Text Chat**: Type your message and get AI responses
            
            **Voice Input**: Upload audio files (WAV format)
            
            **Image Analysis**: Upload images for AI visual analysis
            
            **Multimodal**: Combine text, voice, and images
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 Input Area")
        
        text_input = None
        audio_file = None
        image_file = None
        
        if mode in ["💬 Text Chat", "🌐 Multimodal"]:
            text_input = st.text_area(
                "Enter your message:",
                height=100,
                placeholder="Type your question here..."
            )
        
        if mode in ["🎤 Voice Input", "🌐 Multimodal"]:
            audio_file = st.file_uploader(
                "Upload audio file (WAV format)",
                type=["wav"],
                key="audio"
            )
        
        if mode in ["🖼️ Image Analysis", "🌐 Multimodal"]:
            image_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                key="image"
            )
            
            if image_file:
                # <-- FIXED: Removed use_container_width=True
                st.image(image_file, caption="Uploaded Image")
        
        submit = st.button("🚀 Submit", use_container_width=True)
    
    with col2:
        st.subheader("📊 Status")
        status_container = st.container()
        
        with status_container:
            st.metric("Chat Messages", len(st.session_state.chat_history))
            st.metric("Current Mode", mode.split()[1])
    
    # Process input
    if submit:
        final_query = ""
        response_text = ""
        
        with st.spinner("🤔 Processing your request..."):
            try:
                if audio_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_file.read())
                        tmp_path = tmp_file.name
                    
                    transcribed_text, error = speech_to_text(tmp_path)
                    os.unlink(tmp_path)
                    
                    if error:
                        st.error(f"❌ {error}")
                        st.stop()
                    
                    st.success(f"🎤 Transcribed: {transcribed_text}")
                    final_query = transcribed_text
                
                if text_input:
                    final_query = text_input if not final_query else f"{final_query}. {text_input}"
                
                if image_file:
                    image_bytes = image_file.read()
                    query = final_query or "Describe this image in detail"
                    
                    response_text, error = process_image_with_gemini(image_bytes, query, api_key)
                    
                    if error:
                        st.error(f"❌ {error}")
                        st.stop()
                    else:
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(response_text)
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "user": query,
                            "assistant": response_text
                        })
                        st.success("✅ Image analyzed successfully!")
                elif final_query:
                    if use_agent:
                        agent = get_langchain_agent(api_key)
                        response_text = agent.invoke({"input": final_query})["output"]
                    else:
                        if enable_streaming:
                            response_placeholder = st.empty()
                            full_response = ""
                            
                            for chunk in get_streaming_response(
                                final_query,
                                st.session_state.chat_history,
                                api_key
                            ):
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                            
                            response_placeholder.markdown(full_response)
                            response_text = full_response
                        else:
                            # Non-streaming response
                            # <-- FIXED
                            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                            response = model.generate_content(final_query)
                            response_text = response.text
                
                if final_query and response_text:
                    st.session_state.chat_history.append({
                        "user": final_query,
                        "assistant": response_text
                    })
                    
                    st.success("✅ Response generated successfully!")
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"💭 Message {len(st.session_state.chat_history) - i}", expanded=(i==0)):
                st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{chat["user"]}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br>{chat["assistant"]}</div>', 
                           unsafe_allow_html=True)

if __name__ == "__main__":
    main()