import streamlit as st
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")

# Updated movie dataset
movies = [
    "A man discovers where reality is an illusion and he joins a resistance to fight a digital overlords. (The Matrix)",
    "An exiled heir returns to take back his homeland from a tyrant uncle. (The Lion King)",
    "A young couple from different worlds fall in love aboard a doomed ship. (Titanic)",
    "A group undertakes a journey to destroy a powerful object and defeat a rising darkness. (The Lord of the Rings)",
    "A young orphan is invited to a hidden institution where he learns to harness mystical forces. (Harry Potter)",
    "A man wakes up with no memory and evades secret agents while uncovering his past. (The Bourne Identity)",
    "A linguist must interpret an alien language to prevent global war. (Arrival)",
    "A baseball manager uses data and algorithms to rebuild his losing team. (Moneyball)",
    "A student builds a tech empire while navigating betrayal and lawsuits. (The Social Network)",
    "A lonely man falls in love with an intelligent operating system. (Her)",
    #"A young man moves to a new city where he is invited for a job and tries to balance his personal and professional life. (The New Beginnings)"
]

movie_titles = [m.split(" (")[1].replace(")", "") for m in movies]

# Streamlit App UI
st.title("🎬 Find the movie!")

st.markdown("""
This app finds the most relevant movie based on a description you provide using **3 similarity methods**:

1. 🧮 **CountVectorizer** – Matches based on **exact word overlap**.
2. 📊 **TF-IDF** – Like Count, but **downweights common words** (e.g., "the", "a").
3. 🧠 **spaCy Embeddings** – Matches by **meaning**, not just words (using semantic similarity).

Try the example below or describe a movie you want to see:
> *"A boy goes to school for magic, where he is taught to control powerful spells and faces a dark force."*


""")

remove_stop = st.checkbox("🔘 Remove stopwords for Count/TF-IDF?", value=True)

query = st.text_input("📝 Describe a movie you'd like to see:")

if query:
    all_texts = movies + [query]

    # Stopword removal (for Count/TF-IDF only)
    if remove_stop:
        from sklearn.feature_extraction import text
        custom_stopwords = text.ENGLISH_STOP_WORDS
        analyzer = CountVectorizer().build_analyzer()
        all_texts_processed = [
            " ".join([w for w in analyzer(doc) if w not in custom_stopwords])
            for doc in all_texts
        ]
    else:
        all_texts_processed = all_texts

    # Define a function to clean words depending on stopword toggle
    def cleaned_words(text):
        if remove_stop:
            return set([w for w in analyzer(text) if w not in custom_stopwords])
        else:
            return set(text.lower().split())

    # Count Vectorizer (Regular Cosine)
    vec_count = CountVectorizer()
    mat_count = vec_count.fit_transform(all_texts_processed)
    sim_count = cosine_similarity(mat_count[-1], mat_count[:-1])
    best_count = np.argmax(sim_count)
    count_words = cleaned_words(query) & cleaned_words(movies[best_count])

    # TF-IDF (Regular Cosine)
    vec_tfidf = TfidfVectorizer()
    mat_tfidf = vec_tfidf.fit_transform(all_texts_processed)
    sim_tfidf = cosine_similarity(mat_tfidf[-1], mat_tfidf[:-1])
    best_tfidf = np.argmax(sim_tfidf)
    tfidf_words = cleaned_words(query) & cleaned_words(movies[best_tfidf])

    # spaCy embeddings (Soft Cosine)
    docs = [nlp(text) for text in all_texts]
    query_doc = docs[-1]
    scores_spacy = [query_doc.similarity(doc) for doc in docs[:-1]]
    best_spacy = np.argmax(scores_spacy)

    # Display Results
    st.markdown("## 🔍 Top Matches")

    # CountVectorizer results
    st.markdown("### 🧮 CountVectorizer")
    st.write(f"**Match:** {movies[best_count]}")
    st.caption("Relies on **exact word overlap** (regular cosine similarity).")
    st.write("**Shared words:**", ", ".join(count_words) if count_words else "No exact match.")

    # TF-IDF results
    st.markdown("### 📊 TF-IDF")
    st.write(f"**Match:** {movies[best_tfidf]}")
    st.caption("Gives **less weight** to common words (regular cosine similarity).")
    st.write("**Shared words:**", ", ".join(tfidf_words) if tfidf_words else "No strong overlap.")

    # spaCy Embeddings results
    st.markdown("### 🧠 spaCy Embeddings")
    st.write(f"**Match:** {movies[best_spacy]}")
    st.caption("Uses **semantic meaning** (soft cosine similarity).")
    
    st.markdown("---")
    
    # Explanation of Regular vs Soft Cosine
    st.subheader("📚 Difference between Regular and Soft Cosine")
    st.write("""
- **Regular Cosine (CountVectorizer & TF-IDF)**: Measures similarity based **on exact word overlap**.  
  - For example, "magic" and "wizard" will not match unless both appear in the input query and the movie description.
  
- **Soft Cosine (spaCy Embeddings)**: Measures similarity based **on the meaning of words**, not just the exact words.  
  - This means words like "magic" and "wizard" will be considered similar, even if they don't overlap exactly.
""")

    # Word Meaning Heatmap (spaCy)
    st.subheader("🧠 Word Meaning Heatmap")

    st.write("""
Below is a heatmap comparing **your keywords** to keywords in the matched movie using spaCy's word vectors.  
Darker = more similar in meaning.
""")

    query_keywords = [token.text for token in query_doc if token.is_alpha and not token.is_stop]
    match_doc = docs[best_spacy]
    match_keywords = [token.text for token in match_doc if token.is_alpha and not token.is_stop]

    similarity_matrix = np.zeros((len(query_keywords), len(match_keywords)))
    for i, q_word in enumerate(query_keywords):
        for j, m_word in enumerate(match_keywords):
            similarity_matrix[i, j] = nlp(q_word).similarity(nlp(m_word))

    if similarity_matrix.size > 0:
        df_sim = pd.DataFrame(similarity_matrix, index=query_keywords, columns=match_keywords)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df_sim, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("Not enough meaningful words to compare for the visualization.")
