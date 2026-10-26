This repository contains a two-part data science project focused on building a Content-Based Movie Recommendation System. The project covers the entire pipeline, from raw data cleaning and feature engineering to calculating cosine similarity for personalized suggestions.

# 🚀 Project Overview
The goal is to recommend movies to users based on a "Content" feature that combines genres and user-generated tags. By analyzing the textual relationships between movies, the system can suggest films similar to those a user already enjoys (e.g., searching for "Star Wars" returns other sci-fi space operas).

# 📊 Repository Structure & Workflow
## 1. Data Cleaning & Preprocessing (cleaned_data.ipynb)
Before building the model, the raw MovieLens dataset is processed to create a unified searchable format.

Data Merging: Combines four datasets: movies.csv, tags.csv, ratings.csv, and links.csv.

Feature Engineering: Creates a content column by merging genres and movie tags, creating a rich textual profile for every film.

Rating Aggregation: Calculates the average rating for each movie to help prioritize high-quality recommendations.

Output: Generates a final cleaned_movies.csv file used by the recommendation engine.

## 2. Movie Recommendation Engine (movie_recommendation.ipynb)
This notebook implements the machine learning logic to find similar movies.

TF-IDF Vectorization: Converts the textual "content" of movies into numerical vectors using TfidfVectorizer.

Cosine Similarity: Computes similarity scores between all movie vectors to find which films are mathematically "closest" to each other.

Search Functionality: A custom search tool that allows users to input a movie title and receive a list of the most similar films based on their themes and genres.

# 🛠️ Tools & Libraries
Python 3

Pandas & NumPy: For data manipulation and merging.

Scikit-learn: For the TfidfVectorizer and cosine_similarity metrics.

## 📈 Example Results
The system is capable of high-level semantic matching. For example:

Input: "Star Wars"

Output: Returns other Action/Adventure/Sci-Fi titles with similar tags like "space opera" and "darth vader."
