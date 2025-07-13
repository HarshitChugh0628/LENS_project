import streamlit as st
import pandas as pd
import joblib

spam_model = joblib.load("spam_classifier.pkl")
language_model = joblib.load("lang_det.pkl")
news_model = joblib.load("news_cat2.pkl")
review_model = joblib.load("review.pkl")


st.set_page_config(page_title="LENSE Expert - NLP Suite", page_icon="🤖", layout="wide")

st.markdown("""<h1 style='text-align:center; color:#FF4B4B;'>🤖 LENSE Expert(NLP Suits)</h1>""", unsafe_allow_html=True)



def local_css(css_code):
    st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

local_css("""
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #FFD700; /* Gold headings */
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
    }
""")

## sidebar
with st.sidebar:
    st.image("sh-2.png", use_container_width=True)
    with st.expander("🧑‍🤝‍🧑 About us"):
        st.write("We are students building smart NLP tools using ML.")
    with st.expander("📞 Contact us"):
        st.write("Email: harshit@example.com")
        st.write("GitHub: github.com/harshit")


## tabs
st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["Spam Classifier","Language Detection","Food Review Sentiment","News Classification"])

with tab1:
    # st.title("📩 Spam Classification")
    st.markdown("""<h2 style='color:yellow'>📩 Spam Classification</h2>""", unsafe_allow_html=True)
    msg = st.text_input("Enter Msg ")
    if st.button("Prediction"):
        pred = spam_model.predict([msg])
        if pred[0]==0:
            st.image("spam.png")
        else:
            st.image("ham.jpeg")
            st.balloons()
            st.snow()
    upload_file1 = st.file_uploader("Choose a file",type=["csv","txt"])
    if upload_file1:
        df_spam = pd.read_csv(upload_file1,header=None,names=['Msg1'])
        df_spam.index = range(1,df_spam.shape[0]+1)
        pred = spam_model.predict(df_spam.Msg1)
        df_spam.index = range(1,df_spam.shape[0]+1)
        df_spam["Prediction"] = pred
        df_spam["Prediction"] = df_spam["Prediction"].map({0:'spam',1:'Not spam'})
        st.dataframe(df_spam)

with tab2:
    # st.title("🌐 Language Detection")
    st.markdown("""<h2 style='color:yellow'>🌐 Language Detection</h2>""", unsafe_allow_html=True)
    msga = st.text_input("Enter Msg")
    if st.button("Prediction "):
        pred = language_model.predict([msga])
        # print(pred)
        st.success(pred[0])
        # if pred[0]=="English":
        #     st.success("English")
        # elif pred[0]=="French":
        #     st.success("French")
        # elif pred[0]=="Spanish":
        #     st.success("Spanish")
        # elif pred[0]=="Portugeese":
        #     st.success("Portugeese")
        # elif pred[0]=="Italian":
        #     st.success("Italian")
        # elif pred[0]=="Russian":
        #     st.success("Russian")
        # elif pred[0]=="Sweedish":
        #     st.success("Sweedish")
        # elif pred[0]=="Malayalam":
        #     st.success("Malayalam")
        # elif pred[0]=="Dutch":
        #     st.success("Dutch")
        # elif pred[0]=="Arabic":
        #     st.success("Arabic")
        # elif pred[0]=="Turkish":
        #     st.success("Turkish")
        # elif pred[0]=="German":
        #     st.success("German")
        # elif pred[0]=="Tamil":
        #     st.success("Tamil")
        # elif pred[0]=="Danish":
        #     st.success("Danish")
        # elif pred[0]=="Kannada":
        #     st.success("Kannada")
        # elif pred[0]=="Greek":
        #     st.success("Greek")
        # elif pred[0]=="Hindi":
        #     st.success("Hindi")
        # else:
        #     st.error("i could not found")

    upload_file2 = st.file_uploader("Choose a file ",type=["csv","txt"])
    if upload_file2:
        df_lang = pd.read_csv(upload_file2,header=None,names=['Msg2'])
        df_lang.index = range(1,df_lang.shape[0]+1)
        pred2 = language_model.predict(df_lang.Msg2)
        df_lang["Prediction"] = pred2
        st.dataframe(df_lang)


with tab3:
    # st.title("🍔 Food Review Sentiment")
    st.markdown("""<h2 style='color:yellow'>🍔 Food Review Sentiment</h2>""", unsafe_allow_html=True)
    msg3 = st.text_input("Enter msg")
    if st.button("prediction"):
        pred = spam_model.predict([msg3])
        if pred[0]==0:
            st.error("Disappointing")
        else:
            st.success("Amazing")
            st.balloons()
            st.snow()

    upload_file3 = st.file_uploader("choose a file",type=["csv","txt"])
    if upload_file3:
        df_food = pd.read_csv(upload_file3,header=None,names=['Msg3'])
        df_food.index = range(1,df_food.shape[0]+1)
        pred = review_model.predict(df_food.Msg3)
        df_food.index = range(1,df_food.shape[0]+1)
        df_food["Prediction"] = pred
        df_food["Prediction"] = df_food["Prediction"].map({0:'Disappointing',1:'Amazing'})
        st.dataframe(df_food)

with tab4:
    # st.title("📰 News Classification")
    st.markdown("""<h2 style='color:yellow'>📰 News Classification</h2>""", unsafe_allow_html=True)
    msg4 = st.text_input(" Enter Msg")
    if st.button(" Prediction "):
        pred = news_model.predict([msg4])
        # print(pred)
        st.success(pred[0])

        # if pred[0]=="CULTURE & ARTS":
        #     st.success("CULTURE & ARTS.png")
        # elif pred[0]=="EDUCATION":
        #     st.success("EDUCATION")
        # elif pred[0]=="LATINO VOICES":
        #     st.success("LATINO VOICES")
        # elif pred[0]=="ENVIRONMENT":
        #     st.success("ENVIRONMENT")
        # elif pred[0]=="SCIENCE":
        #     st.success("SCIENCE")
        # elif pred[0]=="TRAVEL":
        #     st.success("TRAVEL")

        # elif pred[0]=="RELIGION":
        #     st.success("RELIGION")

        # elif pred[0]=="BUSINESS":
        #     st.success("BUSINESS")

        # elif pred[0]=="IMPACT":
        #     st.success("IMPACT")

        # elif pred[0]=="SPORTS":
        #     st.success("SPORTS")
        # elif pred[0]=="TECH":
        #     st.success("TECH")

        # elif pred[0]=="WOMEN":
        #     st.success("WOMEN")
        # elif pred[0]=="MEDIA":
        #     st.success("MEDIA")

        # elif pred[0]=="CRIME":
        #     st.success("CRIME")

        # elif pred[0]=="WEIRD NEWS":
        #     st.success("WEIRD NEWS")
        # elif pred[0]=="QUEER VOICES":
        #     st.success("QUEER VOICES")
        # elif pred[0]=="BLACK VOICES":
        #     st.success("BLACK VOICES")
        # elif pred[0]=="WORLD NEWS":
        #     st.success("WORLD NEWS")
        # elif pred[0]=="COMEDY":
        #     st.success("COMEDY")
        # elif pred[0]=="ENTERTAINMENT":
        #     st.success("ENTERTAINMENT")
        # elif pred[0]=="POLITICS":
        #     st.success("POLITICS")
        # else:
        #     st.error("i could not found")

    upload_file4 = st.file_uploader("Choose a File ",type=["csv","txt"])
    if upload_file2:
        df_news = pd.read_csv(upload_file4,header=None,names=['Msg4'])
        df_news.index = range(1,df_news.shape[0]+1)
        pred = news_model.predict(df_news.Msg4)
        df_news["Prediction"] = pred
        st.dataframe(df_news)

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: transparent;
        color: White;
        text-align: center;
        padding: 10px;
        font-size: 15px;
    }
    </style>
    <div class="footer">
        developed ❤️ by Harshit | BCA Project | 2025
    </div>
""", unsafe_allow_html=True)
