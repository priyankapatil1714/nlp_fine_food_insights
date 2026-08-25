import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="NLP Insights:Amazon Fine Food Reviews", layout="wide")

#Load data (all)
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Priyanka/Datascience/NLP_projects/cleaned_data.csv")
    topics = pd.read_csv("C:/Users/Priyanka/Datascience/NLP_projects/topics.csv")
    topic_sentiment = pd.read_csv("C:/Users/Priyanka/Datascience/NLP_projects/topic_sentiment_insight.csv")

    #Time column = Dataset has unix 'time' col
    df["date"] = pd.to_datetime(df["Time"], unit="s",errors="coerce")
    df["year"] = df["date"].dt.year
    return df, topics, topic_sentiment

df, topics_df, topic_sentiment = load_data()

#Header
st.title("NLP INSIGHTS: Amazon Fine Food Reviews")
st.markdown(
    "End-to-end pipeline: transformer based sentiment analysis + BERTopic topic modelling"
    "on unstructed customer reviews, surfacing what drives negative sentiment."
)

#Headline metrics
col1, col2, col3, col4 =st.columns(4)
col1.metric("Total Reviews Analyzed",f"{len(df):,}")
col2.metric("Negative Rate",f"{(df['sentiment']=='negative').mean():.1%}")
col3.metric("Topics Discovered", f"{topics_df['topic'].nunique() - 1}")
col4.metric("Best Model Accuracy", "81%")

st.divider()

#Tabs
tab1, tab2, tab3= st.tabs(["📊 Sentiment Overview", "🔍 Topic Insights", "📈 Trends Over Time"])


#Tab 1:Sentiment Overview
with tab1:
    col1, col2= st.columns([1,1])
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts= df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns= ["sentiment","count"]
        fig= px.bar(sentiment_counts, x="sentiment", y="count", color="sentiment",
                    color_discrete_map={"positive":"#2ecc71","negative":"#e74c3c","neutral":"#95a5a6"})
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.subheader("Model Comparison")
        st.markdown("Three apporaches were benchmarked on the same reviews:")
        comparison = pd.DataFrame({
            "Model":["TF-IDF + LogReg","Binary Transformer + threshold","Native 3-class Transformer"],
            "Accuracy":[0.71, 0.72, 0.81],
            "Macro F1":[0.55, 0.49, 0.62],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption(
            "Native 3-class model won on every metric. The binary+threshold approach"
            "improved negative recall but broke neutral(recall 0.07)-showing model"
            "architecture matters as much as raw model quality."
        )

#Tab2:Topic insights
with tab2:
    st.subheader("Which Topics Drive Negative Sentiment?")
    display_cols =["topic","negative","neutral","positive","size","top_words"]
    display_cols =[c for c in display_cols if c in topic_sentiment.columns]
    sorted_topics = topic_sentiment.sort_values("negative",ascending=False)
    st.dataframe(
        sorted_topics[display_cols].head(15).style.format({"negative": "{:.1%}", "neutral": "{:.1%}", "positive": "{:.1%}"}),
        use_container_width=True, hide_index=True
    )
    baseline_neg = (df["sentiment"]=="negative").mean()
    st.caption(f"Baseline negative rate across all reviews:{baseline_neg:.1%}. Topics above this line over-index on complaints.")

    st.subheader("Topic Word Cloud")
    selected_topic= st.selectbox("Pick a topic to explore:", sorted_topics["topic"].head(10).tolist())
    topic_words= sorted_topics[sorted_topics["topic"]==selected_topic]["top_words"].values[0]
    wc =WordCloud(width=800, height=300, background_color="white").generate(topic_words.replace(","," "))
    fig, ax = plt.subplots(figsize=(10,3))
    ax.imshow(wc,interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # show a few example reviews from this topic
    st.subheader(f"Sample reviews from Topic{selected_topic}")
    examples = topics_df[topics_df["topic"]==selected_topic][["Text","sentiment"]].head(5)
    for _, row in examples.iterrows():
        st.markdown(F"**[{row['sentiment']}]**{row['Text'][:250]}..")

#Tab3:Trends over time
with tab3:
    st.subheader("Sentiment Trend Over Time")
    yearly= df.groupby(["year","sentiment"]).size().reset_index(name="count")
    yearly_pct= df.groupby("year")["sentiment"].value_counts(normalize=True).rename("pct").reset_index()
    fig2= px.line(yearly_pct, x="year",y="pct",color="sentiment",markers=True,
                  color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
    fig2.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Look for any year where negative % spikes")
    st.subheader("Review Volume Over Time")
    volume= df.groupby("year").size().reset_index(name="count")
    fig3 = px.bar(volume, x="year", y="count")
    st.plotly_chart(fig3, use_container_width=True)
st.divider()
st.caption("Built as an end-to-end NLP pipeline: TF-IDF → transformer sentiment analysis → BERTopic topic modeling → insight generation.")


