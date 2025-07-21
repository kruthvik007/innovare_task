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

## Building and Running the Docker Image Locally

The below are the steps to build the Docker image and run the API locally on your machine:

### Step 1: Make sure Docker is installed and running

Install Docker from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) if you haven't already.

### Step 2: Set up the `.env` file

To load the Mistral-7B model, you need to authenticate with the Hugging Face Hub.

1. Log in to your Hugging Face account at: https://huggingface.co
2. Go to your account settings and generate a **read access token**.
3. Create a `.env` file in the project root and add your token as follows:

`HF_TOKEN=your_huggingface_token_here`

### Step 3: Build the Docker image

Run the following command from the project directory:

```bash
docker build -t fot-api .
```

This command builds the Docker image for the application.

Once the image is built, use the command below to start the container:

```
docker run --env-file .env -p 5000:5000 fot-api
```

You can access the running API using Postman or any command-line tool like `curl` by sending requests to this port as shown above.


## Preferred Deployment Platform: Google Cloud Run

If I were to deploy this API, I would choose **Google Cloud Run**. It's a strong fit for this project for a few key reasons:

- It supports containerized applications natively, so I can deploy the Docker image without having to manage servers or infrastructure manually.
- It automatically scales based on incoming traffic, and it scales down to zero when idle, which makes it cost-efficient—especially useful for an API that may not be in constant use.
- It's fully managed, so I wouldn’t need to handle things like load balancing, networking, or resource provisioning.
- It integrates easily with Vertex AI, which gives me a future-proof path in case I want to offload model inference to a managed ML platform.
- It also provides secure and simple ways to handle environment variables, which is important for managing secrets like the Hugging Face token used in this project.

Overall, Cloud Run offers the right balance of simplicity, flexibility, and scalability for this kind of workload.

















