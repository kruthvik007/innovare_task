# Innovare Task

## Bonus 1: API Microservice Deployment

This is a Flask-based API that recommends the top 3 planning questions from the FOT Toolkit based on a student narrative. It uses semantic search via SentenceTransformers and FAISS, and leverages the Mistral-7B language model from Hugging Face to refine and return the most relevant questions.

## How the API Works

This project runs a Flask server with a single POST endpoint:

- `/recommend`: Accepts a JSON payload with a student narrative and returns the top 3 matched planning questions in JSON format.

Example request:

```bash
curl -X POST http://localhost:5000/recommend \
     -H "Content-Type: application/json" \
     -d '{"narrative": "This student is struggling to keep up with coursework, having failed one core class and earning only 2.5 credits out of 4 expected. Attendance is a concern at 88%, and they had one behavioral incident. The student needs academic and attendance support to get back on track."}'
```

## Testing the API Using Postman

I tried to access and test the API using Postman. The steps I followed are below:

1. Create a **POST** request.
2. Set the request URL to: `http://localhost:5000/recommend`
3. In the **Headers** tab, add:
   - `Key = Content-Type`
   - `Value = application/json`
4. In the **Body** tab, select `raw` and choose `JSON` format.
5. Enter the JSON payload. For example:

```json
{
    "narrative": "This student is struggling to keep up with coursework, having failed one core class and earning only 2.5 credits out of 4 expected. Attendance is a concern at 88%, and they had one behavioral incident. The student needs academic and attendance support to get back on track."
}
```


## Project Structure

```
├── app.py                   # Flask server
├── question_selector.py     # LLM-based semantic question selector
├── questions.txt            # Source question pool
├── FOT-Toolkit_*.txt        # OCR extracted FOT document
├── .env                     # Hugging Face token (not committed)
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Recommended Deployment: Google Cloud Run

Google Cloud Run is the preferred platform for deploying this containerized API due to the following advantages:

- **Seamless container support**: Easy to deploy Docker images without any additional configuration.
- **Automatic scaling**: Instantly scales to handle incoming requests and scales down to zero when idle, reducing costs.
- **Fully managed infrastructure**: No need to manage servers, clusters, or provisioning.
- **Native integration with Vertex AI**: Makes it easy to extend the system with advanced model-serving capabilities when needed.
- **Secure environment variable management**: Google cloud run Effortlessly manage secrets like Hugging Face tokens using Cloud Run’s built-in environment support.

















