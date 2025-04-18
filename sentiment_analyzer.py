# Import the 'pipeline' function from the transformers library
from transformers import pipeline

# Load a pre-trained sentiment analysis model
# The library downloads the model the first time you run this!
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis")
print("Model loaded.")

# --- Define Text to Analyze ---
# Option 1: Simple text string
document_text = "I attended the project meeting today, and it was surprisingly productive. The team alignment seems much better."

# Option 2: (If you want to try later) Read from a file
# try:
#     with open("sample_document.txt", "r") as f:
#         document_text = f.read()
# except FileNotFoundError:
#     print("sample_document.txt not found, using default text.")
#     document_text = "This is a default text because the file was missing. It's okay."


# --- Analyze Sentiment ---
print(f"\nAnalyzing text: '{document_text[:100]}...'") # Show first 100 chars
results = sentiment_pipeline(document_text)

# --- Display Results ---
print("\n--- Sentiment Analysis Results ---")
# The result is usually a list containing a dictionary
if results:
    sentiment = results[0]['label']
    score = results[0]['score']
    print(f"Detected Sentiment: {sentiment}")
    print(f"Confidence Score: {score:.4f}")
else:
    print("Could not determine sentiment.")

print("---------------------------------")

# Example of analyzing multiple sentences (pipeline handles batching)
sentences = [
    "This tutorial is quite clear and helpful.",
    "I am worried about the project deadline.",
    "The weather today is absolutely beautiful!"
]
print("\nAnalyzing multiple sentences:")
batch_results = sentiment_pipeline(sentences)
for sentence, result in zip(sentences, batch_results):
    print(f"- '{sentence}': {result['label']} ({result['score']:.4f})")