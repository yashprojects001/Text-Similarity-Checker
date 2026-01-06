# Text Similarity Checker using NLP

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input texts
text1 = input("Enter first text:\n")
text2 = input("\nEnter second text:\n")

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([text1, text2])

# Compute cosine similarity
similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

# Convert to percentage
similarity_percentage = similarity_score * 100

print("\n🔹 Similarity Result 🔹")
print(f"Similarity Percentage: {similarity_percentage:.2f}%")
