FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend into the container
COPY . .

# Expose port 7860, which is the default for Hugging Face Spaces
EXPOSE 7860

# Command to run the Gunicorn server on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
