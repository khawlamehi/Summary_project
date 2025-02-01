import os
import spacy
import nltk
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import docx
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Téléchargement des ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger le modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Tokenisation des phrases
    sentences = sent_tokenize(text, language='french')
    
    # Tokenisation des mots et prétraitement
    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()
    
    processed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent, language='french')
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
        processed_sentences.append(" ".join(words))
    
    return sentences, processed_sentences

# Fonction de génération de résumé avec spaCy
def summarize_with_spacy(sentences, processed_sentences, top_n=2):
    # Calculer les embeddings des phrases
    sentence_embeddings = [nlp(sent).vector for sent in processed_sentences]
    
    # Calculer l'embedding moyen du document
    doc_embedding = np.mean(sentence_embeddings, axis=0)
    
    # Calculer la similarité cosinus entre chaque phrase et le document
    similarities = []
    for sent_embedding in sentence_embeddings:
        similarity = np.dot(sent_embedding, doc_embedding) / (np.linalg.norm(sent_embedding) * np.linalg.norm(doc_embedding))
        similarities.append(similarity)
    
    # Sélectionner les phrases les plus importantes
    top_sentence_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Retourner le résumé
    summary = [sentences[i] for i in top_sentence_indices]
    return summary

@csrf_exempt
def summarize(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        file = request.FILES.get('file', None)
        
        # Si un fichier est téléchargé
        if file:
            ext = os.path.splitext(file.name)[1]
            if ext == '.txt':
                text = file.read().decode('utf-8')
            elif ext == '.docx':
                doc = docx.Document(file)
                text = '\n'.join([para.text for para in doc.paragraphs])
        
        # Si aucun texte ou fichier n'est fourni, on renvoie une erreur
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        # Prétraitement du texte et génération du résumé
        original_sentences, processed_sentences = preprocess_text(text)
        summary = summarize_with_spacy(original_sentences, processed_sentences, top_n=2)
        
        # Retour du résumé sous forme JSON
        return JsonResponse({'summary': ' '.join(summary)})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Page d'accueil
def index(request):
    return render(request, 'index.html')
