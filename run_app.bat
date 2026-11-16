@echo off
title RAG Retrieval Lab
cd /d "%~dp0"
echo Starting the RAG Retrieval Lab...
echo Open the URL printed below (usually http://localhost:8501)
".venv\Scripts\python.exe" -m streamlit run "streamlit_app\app.py"
pause
